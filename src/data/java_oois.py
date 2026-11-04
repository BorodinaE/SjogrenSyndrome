"""Minimal Java ObjectInputStream walker for QuPath .qpdata files.

QuPath serialises its data with the Java Object Serialization Protocol
(JDK 1.5+). Off-the-shelf parsers (javaobj, javaobj.v2) cannot consume the
full stream because several QuPath classes use custom `writeObject` methods
that emit application-specific bytes.

This walker only goes deep enough to:
    * track the global handle counter (Java assigns handles sequentially
      starting at 0x7e0000 to every TC_OBJECT, TC_CLASSDESC, TC_STRING,
      TC_LONGSTRING, TC_ARRAY, TC_ENUM, TC_PROXYCLASSDESC)
    * decode TC_STRING / TC_LONGSTRING values
    * decode byte[] (`[B`) array contents — that's where WKB geometries live
    * resolve TC_REFERENCE back to the previously assigned handle
    * read all the standard primitive/object fields of every class so the
      stream cursor stays aligned, even for QuPath's custom classes (the
      `writeObject` annotation is consumed as raw content until
      TC_ENDBLOCKDATA)

The walker stays untyped at the value level: it surfaces a flat event list
that downstream code can scan to associate each WKB byte array with the
nearest preceding PathClass handle. That is sufficient to recover all
geojson from a QuPath .qpdata file, including polygons that share a
class with a previously-serialised annotation.

Spec reference: https://docs.oracle.com/javase/8/docs/platform/serialization/spec/protocol.html
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

# Stream constants (Java Object Serialization Stream Protocol).
STREAM_MAGIC = 0xACED
STREAM_VERSION = 5

TC_NULL = 0x70
TC_REFERENCE = 0x71
TC_CLASSDESC = 0x72
TC_OBJECT = 0x73
TC_STRING = 0x74
TC_ARRAY = 0x75
TC_CLASS = 0x76
TC_BLOCKDATA = 0x77
TC_ENDBLOCKDATA = 0x78
TC_RESET = 0x79
TC_BLOCKDATALONG = 0x7A
TC_EXCEPTION = 0x7B
TC_LONGSTRING = 0x7C
TC_PROXYCLASSDESC = 0x7D
TC_ENUM = 0x7E

# Class descriptor flag bits.
SC_WRITE_METHOD = 0x01
SC_BLOCK_DATA = 0x08
SC_SERIALIZABLE = 0x02
SC_EXTERNALIZABLE = 0x04
SC_ENUM = 0x10

BASE_HANDLE = 0x7E0000

# Primitive numeric array types whose raw bytes we want to preserve in the
# event stream (rather than just skipping the cursor). Used by PolygonROI for
# X/Y coordinates (double[]) and by measurement lists (float[]). Maps the
# Java type signature → bytes per element.
_PRIMITIVE_NUMERIC_ARRAYS: dict[str, int] = {
    "[F": 4,  # float[]
    "[D": 8,  # double[]
    "[I": 4,  # int[]
    "[J": 8,  # long[]
    "[S": 2,  # short[]
}


@dataclass
class _ClassDesc:
    name: str
    flags: int
    fields: list[tuple[str, str]]  # (type_code, field_name); object types carry "L...;" / "[..."
    super_desc: "_ClassDesc | None"

    def all_fields(self) -> list[tuple[str, str, str]]:
        """Fields walked in super→this order, tagged with owning class name."""
        out: list[tuple[str, str, str]] = []
        if self.super_desc is not None:
            out.extend(self.super_desc.all_fields())
        for type_code, fname in self.fields:
            out.append((self.name, type_code, fname))
        return out

    def chain(self) -> list["_ClassDesc"]:
        """super-most → most-specific class descriptors."""
        chain: list[_ClassDesc] = []
        if self.super_desc is not None:
            chain.extend(self.super_desc.chain())
        chain.append(self)
        return chain


@dataclass
class StreamEvent:
    """One handle-bearing or otherwise notable token in the stream."""
    offset: int
    handle: int | None  # None for TC_NULL / TC_REFERENCE / TC_ENDBLOCKDATA
    kind: str           # "string" | "bytearray" | "object" | "classdesc" | "reference" | "null"
    payload: Any = None # str for strings, bytes for bytearrays, classdesc name for objects/classdescs, target handle for references


class _Cursor:
    """Bytes cursor with primitive readers for Java DataInput semantics (big-endian)."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def peek_u8(self) -> int:
        return self.data[self.pos]

    def read_u8(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b

    def read_u16(self) -> int:
        v = struct.unpack_from(">H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_i16(self) -> int:
        v = struct.unpack_from(">h", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_i32(self) -> int:
        v = struct.unpack_from(">i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_u32(self) -> int:
        v = struct.unpack_from(">I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_i64(self) -> int:
        v = struct.unpack_from(">q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def read_f32(self) -> float:
        v = struct.unpack_from(">f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_f64(self) -> float:
        v = struct.unpack_from(">d", self.data, self.pos)[0]
        self.pos += 8
        return v

    def read_bytes(self, n: int) -> bytes:
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def read_utf(self) -> str:
        n = self.read_u16()
        return self.read_bytes(n).decode("utf-8", errors="replace")

    def read_long_utf(self) -> str:
        n = self.read_i64()
        return self.read_bytes(n).decode("utf-8", errors="replace")


class JavaStreamWalker:
    """Walks a Java OIS stream and emits an ordered list of StreamEvent.

    The walker is deliberately permissive: if it cannot interpret a section,
    it raises and the caller can decide to truncate the event list.
    """

    def __init__(self, data: bytes, start: int = 0) -> None:
        self.cur = _Cursor(data)
        self.cur.pos = start
        self.handles: dict[int, StreamEvent] = {}
        self.events: list[StreamEvent] = []
        self._next_handle = BASE_HANDLE

    def _alloc_handle(self, event: StreamEvent) -> int:
        h = self._next_handle
        self._next_handle += 1
        self.handles[h] = event
        event.handle = h
        return h

    # ----- top-level walk ------------------------------------------------

    def walk(self) -> list[StreamEvent]:
        magic = self.cur.read_u16()
        version = self.cur.read_u16()
        if magic != STREAM_MAGIC or version != STREAM_VERSION:
            raise ValueError(f"Bad Java stream header: 0x{magic:04x} v{version}")

        # QuPath truncates the very last writeObject block at EOF rather than
        # closing it with TC_ENDBLOCKDATA. Treat any IndexError/EOFError once
        # the cursor has reached the tail of the stream as graceful end —
        # everything before that has been parsed correctly.
        try:
            while not self.cur.eof():
                self._read_content(allow_blockdata=True)
        except (IndexError, EOFError):
            pass
        return self.events

    # ----- content dispatch ---------------------------------------------

    def _read_content(self, allow_blockdata: bool) -> Any:
        code = self.cur.peek_u8()

        if code == TC_BLOCKDATA or code == TC_BLOCKDATALONG:
            if not allow_blockdata:
                raise ValueError(f"Unexpected blockdata at {self.cur.pos}")
            self._read_blockdata()
            return None
        if code == TC_ENDBLOCKDATA:
            # Caller (custom-write annotation reader) handles this; do not consume here.
            return _END_BLOCK
        return self._read_object()

    def _read_blockdata(self) -> None:
        code = self.cur.read_u8()
        if code == TC_BLOCKDATA:
            n = self.cur.read_u8()
        else:
            n = self.cur.read_u32()
        self.cur.pos += n  # skip raw bytes

    def _read_object(self) -> Any:
        offset = self.cur.pos
        code = self.cur.read_u8()

        if code == TC_NULL:
            self.events.append(StreamEvent(offset=offset, handle=None, kind="null"))
            return None
        if code == TC_REFERENCE:
            target = self.cur.read_i32() & 0xFFFFFFFF
            ev = StreamEvent(offset=offset, handle=None, kind="reference", payload=target)
            self.events.append(ev)
            return self.handles.get(target)
        if code == TC_STRING:
            value = self.cur.read_utf()
            ev = StreamEvent(offset=offset, handle=None, kind="string", payload=value)
            self._alloc_handle(ev)
            self.events.append(ev)
            return ev
        if code == TC_LONGSTRING:
            value = self.cur.read_long_utf()
            ev = StreamEvent(offset=offset, handle=None, kind="string", payload=value)
            self._alloc_handle(ev)
            self.events.append(ev)
            return ev
        if code == TC_CLASSDESC:
            return self._read_new_class_desc(offset)
        if code == TC_PROXYCLASSDESC:
            return self._read_proxy_class_desc(offset)
        if code == TC_CLASS:
            # A Class object: classDesc reference, allocates its own handle.
            cd = self._read_class_desc_field()
            ev = StreamEvent(offset=offset, handle=None, kind="class", payload=cd)
            self._alloc_handle(ev)
            self.events.append(ev)
            return ev
        if code == TC_OBJECT:
            return self._read_new_object(offset)
        if code == TC_ARRAY:
            return self._read_new_array(offset)
        if code == TC_ENUM:
            return self._read_new_enum(offset)
        if code == TC_RESET:
            self.handles.clear()
            self._next_handle = BASE_HANDLE
            return None

        raise ValueError(f"Unknown TC code 0x{code:02x} at offset {offset}")

    # ----- class descriptors --------------------------------------------

    def _read_class_desc_field(self) -> _ClassDesc | None:
        """Read the classDesc reference at the start of newObject/newArray/etc."""
        code = self.cur.peek_u8()
        offset = self.cur.pos
        if code == TC_NULL:
            self.cur.read_u8()
            self.events.append(StreamEvent(offset=offset, handle=None, kind="null"))
            return None
        if code == TC_REFERENCE:
            self.cur.read_u8()
            target = self.cur.read_i32() & 0xFFFFFFFF
            ev = StreamEvent(offset=offset, handle=None, kind="reference", payload=target)
            self.events.append(ev)
            resolved = self.handles.get(target)
            return getattr(resolved, "_classdesc", None) if resolved else None
        if code == TC_CLASSDESC:
            self.cur.read_u8()
            return self._read_new_class_desc(offset)
        if code == TC_PROXYCLASSDESC:
            self.cur.read_u8()
            return self._read_proxy_class_desc(offset)
        raise ValueError(f"Expected classDesc at {offset}, got 0x{code:02x}")

    def _read_new_class_desc(self, offset: int) -> _ClassDesc:
        name = self.cur.read_utf()
        _suid = self.cur.read_i64()  # noqa: F841
        ev = StreamEvent(offset=offset, handle=None, kind="classdesc", payload=name)
        self._alloc_handle(ev)
        self.events.append(ev)

        flags = self.cur.read_u8()
        n_fields = self.cur.read_i16()
        fields: list[tuple[str, str]] = []
        for _ in range(n_fields):
            type_code = chr(self.cur.read_u8())
            fname = self.cur.read_utf()
            if type_code in ("L", "["):
                # Object/array field: followed by a classname-string descriptor.
                desc_string = self._read_object()
                desc_value = (
                    desc_string.payload
                    if isinstance(desc_string, StreamEvent) and desc_string.kind == "string"
                    else ""
                )
                fields.append((type_code + (desc_value or ""), fname))
            else:
                fields.append((type_code, fname))

        # classAnnotation (objects until TC_ENDBLOCKDATA) — ignore content but consume.
        self._read_until_endblock()
        super_desc = self._read_class_desc_field()

        cd = _ClassDesc(name=name, flags=flags, fields=fields, super_desc=super_desc)
        ev._classdesc = cd  # type: ignore[attr-defined]
        ev.payload = cd
        return cd

    def _read_proxy_class_desc(self, offset: int) -> _ClassDesc:
        ev = StreamEvent(offset=offset, handle=None, kind="classdesc", payload="<proxy>")
        self._alloc_handle(ev)
        self.events.append(ev)
        n_ifaces = self.cur.read_i32()
        for _ in range(n_ifaces):
            self.cur.read_utf()
        self._read_until_endblock()
        super_desc = self._read_class_desc_field()
        cd = _ClassDesc(name="<proxy>", flags=0, fields=[], super_desc=super_desc)
        ev._classdesc = cd  # type: ignore[attr-defined]
        ev.payload = cd
        return cd

    def _read_until_endblock(self) -> None:
        """Consume an annotation block: contents until TC_ENDBLOCKDATA.

        QuPath's last `writeObject` block at the very end of the stream is
        truncated (no TC_ENDBLOCKDATA before EOF). Treat EOF here as benign
        rather than raising — the walker has already consumed every object
        of interest by that point.
        """
        while not self.cur.eof():
            code = self.cur.peek_u8()
            if code == TC_ENDBLOCKDATA:
                self.cur.read_u8()
                return
            self._read_content(allow_blockdata=True)

    # ----- objects, arrays, enums ---------------------------------------

    def _read_new_object(self, offset: int) -> StreamEvent:
        cd = self._read_class_desc_field()
        cls_name = cd.name if cd is not None else "<null>"
        ev = StreamEvent(offset=offset, handle=None, kind="object", payload=cls_name)
        self._alloc_handle(ev)
        ev._classdesc = cd  # type: ignore[attr-defined]
        self.events.append(ev)

        if cd is not None:
            # Capture the boxed primitive value for java.lang.Integer (and
            # other simple boxed types) so callers can recover e.g. PathClass
            # colorRGB without re-walking the byte stream. The payload becomes
            # a tuple (class_name, value) when applicable.
            if cls_name == "java.lang.Integer":
                value = self.cur.read_i32()
                ev.payload = (cls_name, value)
            elif cls_name == "java.lang.Long":
                value = self.cur.read_i64()
                ev.payload = (cls_name, value)
            elif cls_name == "java.lang.Boolean":
                value = bool(self.cur.read_u8())
                ev.payload = (cls_name, value)
            else:
                self._read_class_data(cd)
        return ev

    def _read_new_array(self, offset: int) -> StreamEvent:
        cd = self._read_class_desc_field()
        cls_name = cd.name if cd is not None else "<null>"
        ev = StreamEvent(offset=offset, handle=None, kind="bytearray" if cls_name == "[B" else "array",
                         payload=cls_name)
        self._alloc_handle(ev)
        self.events.append(ev)
        size = self.cur.read_i32()
        if cls_name == "[B":
            data = self.cur.read_bytes(size)
            ev.payload = data
        elif cls_name in _PRIMITIVE_NUMERIC_ARRAYS:
            # float[] / double[] / int[] / long[] / short[] — keep raw bytes so
            # downstream code can decode (PolygonROI stores X/Y as double[]).
            element_size = _PRIMITIVE_NUMERIC_ARRAYS[cls_name]
            raw = self.cur.read_bytes(size * element_size)
            ev.payload = (cls_name, size, raw)
        else:
            # Read primitive arrays as raw, object arrays as content sequence.
            element = cls_name[1:] if cls_name.startswith("[") else "L"
            self._read_array_elements(element, size)
        return ev

    def _read_array_elements(self, element: str, size: int) -> None:
        if element == "B" or element == "Z":
            self.cur.pos += size
        elif element == "C" or element == "S":
            self.cur.pos += size * 2
        elif element == "I" or element == "F":
            self.cur.pos += size * 4
        elif element == "J" or element == "D":
            self.cur.pos += size * 8
        else:
            # Object/array of objects
            for _ in range(size):
                self._read_content(allow_blockdata=False)

    def _read_new_enum(self, offset: int) -> StreamEvent:
        cd = self._read_class_desc_field()
        ev = StreamEvent(offset=offset, handle=None, kind="enum",
                         payload=cd.name if cd is not None else "<null>")
        self._alloc_handle(ev)
        self.events.append(ev)
        # enumConstantName
        self._read_object()
        return ev

    # ----- class data ----------------------------------------------------

    def _read_class_data(self, cd: _ClassDesc) -> None:
        """Read data for an object according to its classdesc chain (super → this)."""
        for layer in cd.chain():
            if layer.flags & SC_EXTERNALIZABLE:
                if layer.flags & SC_BLOCK_DATA:
                    self._read_until_endblock()
                else:
                    # Externalizable without block data: cannot generically parse.
                    raise ValueError(f"Non-block-data Externalizable for {layer.name}")
                continue

            # Serializable: read declared fields, then optional writeObject annotation.
            self._read_fields(layer.fields)
            if layer.flags & SC_WRITE_METHOD:
                self._read_until_endblock()

    def _read_fields(self, fields: list[tuple[str, str]]) -> None:
        for type_code, _fname in fields:
            t = type_code[0]
            if t == "B":
                self.cur.read_u8()
            elif t == "C":
                self.cur.read_u16()
            elif t == "D":
                self.cur.read_f64()
            elif t == "F":
                self.cur.read_f32()
            elif t == "I":
                self.cur.read_i32()
            elif t == "J":
                self.cur.read_i64()
            elif t == "S":
                self.cur.read_i16()
            elif t == "Z":
                self.cur.read_u8()
            elif t == "L" or t == "[":
                # Object reference — content with no blockdata.
                self._read_content(allow_blockdata=False)
            else:
                raise ValueError(f"Unknown field type {type_code!r}")


# Sentinel returned by _read_content when it consumed a TC_ENDBLOCKDATA.
_END_BLOCK = object()


def walk_stream(data: bytes) -> list[StreamEvent]:
    """Walk a complete Java OIS stream and return all events in order."""
    return JavaStreamWalker(data).walk()


# ---------------------------------------------------------------------------
# Higher-level extraction tailored to QuPath's PathAnnotationObject layout.
# ---------------------------------------------------------------------------

_PATHCLASS_NAME = "qupath.lib.objects.classes.PathClass"
_PATH_ANNOTATION_NAME = "qupath.lib.objects.PathAnnotationObject"


@dataclass(frozen=True)
class RawAnnotation:
    """A single annotation extracted from the Java stream.

    `class_name` is None when the PathClass field was null (unclassified).
    """
    class_name: str | None
    wkb: bytes
    offset: int


def _resolve_pathclass_names(events: list[StreamEvent]) -> dict[int, str]:
    """Walk events and build (PathClass object handle) → name (string).

    For each new `qupath.lib.objects.classes.PathClass` TC_OBJECT, the field
    layout is (colorRGB: Integer, name: String, parentClass: PathClass).
    The first string event after the PathClass object is the class name.
    Subsequent geojson that reuse the same class refer back to this
    PathClass handle via TC_REFERENCE.
    """
    mapping: dict[int, str] = {}
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.kind == "object" and ev.payload == _PATHCLASS_NAME and ev.handle is not None:
            # Scan forward a bounded window for the first string event — that
            # is the `name` field of the PathClass (colorRGB precedes it as
            # an Integer object, which our walker emits as a "object" event).
            name = None
            for j in range(i + 1, min(i + 30, len(events))):
                if events[j].kind == "string":
                    name = events[j].payload
                    break
                if events[j].kind == "object" and events[j].payload == _PATH_ANNOTATION_NAME:
                    break
            if name is not None:
                mapping[ev.handle] = name
        i += 1
    return mapping


def extract_annotations(data: bytes) -> list[RawAnnotation]:
    """Extract every (class_name, wkb_bytes) pair from a .qpdata stream.

    Recovers geojson that share a PathClass with an earlier annotation by
    resolving the Java TC_REFERENCE handle back to the PathClass object the
    walker recorded on its first appearance.
    """
    events = walk_stream(data)
    handle_to_class = _resolve_pathclass_names(events)

    annotations: list[RawAnnotation] = []
    current_class: str | None = None
    in_annotation = False

    for ev in events:
        if ev.kind == "object" and ev.payload == _PATH_ANNOTATION_NAME:
            in_annotation = True
            current_class = None
            continue
        if not in_annotation:
            continue

        # New PathClass object inside the annotation — class is the next string.
        if ev.kind == "object" and ev.payload == _PATHCLASS_NAME:
            current_class = handle_to_class.get(ev.handle or -1)
            continue

        # Reference to a previously-serialised PathClass object.
        if ev.kind == "reference" and ev.payload in handle_to_class:
            current_class = handle_to_class[ev.payload]
            continue

        # The WKB byte[] is the only sizable byte array inside an annotation;
        # everything else (UUID, metadata strings) is on a different code path.
        if ev.kind == "bytearray" and isinstance(ev.payload, bytes) and len(ev.payload) >= 40:
            annotations.append(RawAnnotation(
                class_name=current_class,
                wkb=ev.payload,
                offset=ev.offset,
            ))
            in_annotation = False  # one WKB per annotation
            current_class = None

    return annotations
