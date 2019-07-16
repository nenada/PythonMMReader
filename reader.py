"""
Library for reading multiresolution micro-magellan
"""
import os
import mmap
import numpy as np
import sys
import json
import platform
import dask.array as da


class _MagellanMultipageTiffReader:
    # Class corresponsing to a single multipage tiff file in a Micro-Magellan dataset. Pass the full path of the TIFF to
    # instantiate and call close() when finished
    # TIFF constants
    WIDTH = 256
    HEIGHT = 257
    BITS_PER_SAMPLE = 258
    COMPRESSION = 259
    PHOTOMETRIC_INTERPRETATION = 262
    IMAGE_DESCRIPTION = 270
    STRIP_OFFSETS = 273
    SAMPLES_PER_PIXEL = 277
    ROWS_PER_STRIP = 278
    STRIP_BYTE_COUNTS = 279
    X_RESOLUTION = 282
    Y_RESOLUTION = 283
    RESOLUTION_UNIT = 296
    MM_METADATA = 51123

    # file format constants
    INDEX_MAP_OFFSET_HEADER = 54773648
    INDEX_MAP_HEADER = 3453623
    SUMMARY_MD_HEADER = 2355492

    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.file = open(tiff_path, 'rb')
        if platform.system() == 'Windows':
            self.mmap_file = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.mmap_file = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.summary_md, self.index_tree, self.first_ifd_offset = self._read_header()
        # get important metadata fields
        self.width = self.summary_md['Width']
        self.height = self.summary_md['Height']

    def close(self):
        self.mmap_file.close()
        self.file.close()

    def _read_header(self):
        """
        :param file:
        :return: dictionary with summary metadata, nested dictionary of byte offsets of TIFF Image File Directories with
        keys [channel_index][z_index][frame_index][position_index], int byte offset of first image IFD
        """
        # read standard tiff header
        if self.mmap_file[:2] == b'\x4d\x4d':
            # Big endian
            if sys.byteorder != 'big':
                raise Exception("Potential issue with mismatched endian-ness")
        elif self.mmap_file[:2] == b'\x49\x49':
            # little endian
            if sys.byteorder != 'little':
                raise Exception("Potential issue with mismatched endian-ness")
        else:
            raise Exception('Endian type not specified correctly')
        if np.frombuffer(self.mmap_file[2:4], dtype=np.uint16)[0] != 42:
            raise Exception('Tiff magic 42 missing')
        first_ifd_offset = np.frombuffer(self.mmap_file[4:8], dtype=np.uint32)[0]

        # read custom stuff: summary md, index map
        index_map_offset_header, index_map_offset = np.frombuffer(self.mmap_file[8:16], dtype=np.uint32)
        if index_map_offset_header != self.INDEX_MAP_OFFSET_HEADER:
            raise Exception('Index map offset header wrong')
        summary_md_header, summary_md_length = np.frombuffer(self.mmap_file[32:40], dtype=np.uint32)
        if summary_md_header != self.SUMMARY_MD_HEADER:
            raise Exception('Index map offset header wrong')
        summary_md = json.loads(self.mmap_file[40:40 + summary_md_length])
        index_map_header, index_map_length = np.frombuffer(
            self.mmap_file[40 + summary_md_length:48 + summary_md_length],
            dtype=np.uint32)
        if index_map_header != self.INDEX_MAP_HEADER:
            raise Exception('Index map header incorrect')
        # get index map as nested list of ints
        index_map_keys = np.array([[int(cztp) for i, cztp in enumerate(entry) if i < 4]
                                   for entry in np.reshape(np.frombuffer(self.mmap_file[48 + summary_md_length:48 +
                                    summary_md_length + index_map_length * 20], dtype=np.int32), [-1, 5])])
        index_map_byte_offsets = np.array([[int(offset) for i, offset in enumerate(entry) if i == 4] for entry in
            np.reshape(np.frombuffer(self.mmap_file[48 + summary_md_length:48 + summary_md_length + index_map_length *
                            20], dtype=np.uint32), [-1, 5])])
        index_map = np.concatenate((index_map_keys, index_map_byte_offsets), axis=1)
        string_key_index_map = {'_'.join([str(ind) for ind in entry[:4]]): entry[4] for entry in index_map}
        # unpack into a tree (i.e. nested dicts)
        index_tree = {}
        for c_index in set([line[0] for line in index_map]):
            for z_index in set([line[1] for line in index_map]):
                for t_index in set([line[2] for line in index_map]):
                    for p_index in set([line[3] for line in index_map]):
                        if '_'.join([str(c_index), str(z_index), str(t_index),
                                     str(p_index)]) in string_key_index_map.keys():
                            # fill out tree as needed
                            if c_index not in index_tree.keys():
                                index_tree[c_index] = {}
                            if z_index not in index_tree[c_index].keys():
                                index_tree[c_index][z_index] = {}
                            if t_index not in index_tree[c_index][z_index].keys():
                                index_tree[c_index][z_index][t_index] = {}
                            index_tree[c_index][z_index][t_index][p_index] = string_key_index_map[
                                '_'.join([str(c_index), str(z_index), str(t_index), str(p_index)])]
        return summary_md, index_tree, first_ifd_offset

    def _read(self, start, end):
        """
        Convert to python ints
        """
        return self.mmap_file[int(start):int(end)]

    def _read_ifd(self, byte_offset):
        """
        Read image file directory. First two bytes are number of entries (n), next n*12 bytes are individual IFDs, final 4
        bytes are next IFD offset location
        :return: dictionary with fields needed for reading
        """
        num_entries = np.frombuffer(self._read(byte_offset, byte_offset + 2), dtype=np.uint16)[0]
        info = {}
        for i in range(num_entries):
            tag, type = np.frombuffer(self._read(byte_offset + 2 + i * 12, byte_offset + 2 + i * 12 + 4),
                                      dtype=np.uint16)
            count = \
            np.frombuffer(self._read(byte_offset + 2 + i * 12 + 4, byte_offset + 2 + i * 12 + 8), dtype=np.uint32)[0]
            if type == 3 and count == 1:
                value = \
                np.frombuffer(self._read(byte_offset + 2 + i * 12 + 8, byte_offset + 2 + i * 12 + 10), dtype=np.uint16)[
                    0]
            else:
                value = \
                np.frombuffer(self._read(byte_offset + 2 + i * 12 + 8, byte_offset + 2 + i * 12 + 12), dtype=np.uint32)[
                    0]
            # save important tags for reading images
            if tag == self.MM_METADATA:
                info['md_offset'] = value
                info['md_length'] = count
            elif tag == self.STRIP_OFFSETS:
                info['pixel_offset'] = value
            elif tag == self.STRIP_BYTE_COUNTS:
                info['bytes_per_image'] = value
        info['next_ifd_offset'] = np.frombuffer(self._read(byte_offset + num_entries * 12 + 2,
                                                           byte_offset + num_entries * 12 + 6), dtype=np.uint32)[0]
        if 'bytes_per_image' not in info or 'pixel_offset' not in info:
            raise Exception('Missing tags in IFD entry, file may be corrupted')
        return info

    def _read_pixels(self, offset, length, memmapped):
        if self.width * self.height * 2 == length:
            pixel_type = np.uint16
        elif self.width * self.height == length:
            pixel_type = np.uint8
        else:
            raise Exception('Unknown pixel type')

        if memmapped:
            return np.memmap(open(self.tiff_path, 'rb'),
                             dtype=pixel_type, mode='r', offset=offset, shape=(self.height, self.width))
        else:
            pixels = np.frombuffer(self._read(offset, offset + length), dtype=pixel_type)
            return np.reshape(pixels, [self.height, self.width])

    def read_metadata(self, channel_index, z_index, t_index, pos_index):
        ifd_offset = self.index_tree[channel_index][z_index][t_index][pos_index]
        ifd_data = self._read_ifd(ifd_offset)
        metadata = json.loads(self._read(ifd_data['md_offset'], ifd_data['md_offset'] + ifd_data['md_length']))
        return metadata

    def read_image(self, channel_index, z_index, t_index, pos_index, read_metadata=False, memmapped=False):
        ifd_offset = self.index_tree[channel_index][z_index][t_index][pos_index]
        ifd_data = self._read_ifd(ifd_offset)
        image = self._read_pixels(ifd_data['pixel_offset'], ifd_data['bytes_per_image'], memmapped)
        if read_metadata:
            bs = self._read(ifd_data['md_offset'], ifd_data['md_offset'] + ifd_data['md_length'])
            #seems like the last byte is always a 0... not sure why
            bs = bs[:-1]
            metadata = json.loads(bs)
            return image, metadata
        return image

    def check_ifd(self, channel_index, z_index, t_index, pos_index):
        ifd_offset = self.index_tree[channel_index][z_index][t_index][pos_index]
        try:
            ifd_data = self._read_ifd(ifd_offset)
            return True
        except:
            return False

class MicroManagerStack:

    def __init__(self, path):
        """
        open all tiff files in directory, keep them in a list, and a tree based on image indices
        :param path:
        """
        tiff_names = [os.path.join(path, tiff) for tiff in os.listdir(path) if tiff.endswith('.tif')]
        self.reader_list = []
        self.reader_tree = {}
        #populate list of readers and tree mapping indices to readers
        for tiff in tiff_names:
            reader = _MagellanMultipageTiffReader(tiff)
            self.reader_list.append(reader)
            it = reader.index_tree
            for c in it.keys():
                if c not in self.reader_tree.keys():
                    self.reader_tree[c] = {}
                for z in it[c].keys():
                    if z not in self.reader_tree[c].keys():
                        self.reader_tree[c][z] = {}
                    for t in it[c][z].keys():
                        if t not in self.reader_tree[c][z].keys():
                            self.reader_tree[c][z][t] = {}
                        for p in it[c][z][t].keys():
                            self.reader_tree[c][z][t][p] = reader

    def read_image(self, channel_index=0, z_index=0, t_index=0, pos_index=0, read_metadata=False, memmapped=False):
        # determine which reader contains the image
        reader = self.reader_tree[channel_index][z_index][t_index][pos_index]
        return reader.read_image(channel_index, z_index, t_index, pos_index, read_metadata, memmapped)

    def read_metadata(self, channel_index=0, z_index=0, t_index=0, pos_index=0):
        # determine which reader contains the image
        reader = self.reader_tree[channel_index][z_index][t_index][pos_index]
        return reader.read_metadata(channel_index, z_index, t_index, pos_index)

    def check_ifd(self, channel_index=0, z_index=0, t_index=0, pos_index=0):
        # determine which reader contains the image
        reader = self.reader_tree[channel_index][z_index][t_index][pos_index]
        return reader.check_ifd(channel_index, z_index, t_index, pos_index)

    def close(self):
        for reader in self.reader_list:
            reader.close()
