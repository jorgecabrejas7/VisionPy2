import struct


def read_bin_file_field_uint64(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 8  # Size of uint64 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of uint64 elements
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}Q", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]

    return data, error


def read_bin_file_field_uint32(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 4  # Size of uint32 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of uint32 elementsa
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}I", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]

    return data, error


def read_bin_file_field_uint16(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 2  # Size of uint16 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of uint16 elements
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}H", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]

    return data, error


def read_bin_file_field_string(f):
    print(
        f"Current file position before reading: {f.tell()}"
    )  # Add this to check file position
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 1  # Size of char data in bytes

    error = 0
    data = ""

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    n_data_file = struct.unpack("I", f.read(4))[0]
    data_bytes = f.read(n_data_file)  # Read the actual characters

    try:
        data = data_bytes.decode("utf-8")
    except UnicodeDecodeError:
        print("Error: Could not decode the string data as UTF-8.")
        return None, -1

    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    return data, error


def read_bin_file_field_int32(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 4  # Size of int32 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of int32 elements
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}i", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]

    return data, error


def read_bin_file_field_int16(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 2  # Size of int16 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of int16 elements
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}h", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]
    print(f"[INFO] --- {n_data_file} {data_size}")
    return data, error


def read_bin_file_field_float64(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 8  # Size of float64 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of float64 elements
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}d", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]

    return data, error


def read_bin_file_field_float32(f):
    APPBIN_iCAMPO = ord("[")  # Start marker
    APPBIN_fCAMPO = ord("]")  # End marker
    DATA_SIZE = 4  # Size of float32 data in bytes

    error = 0
    data = 0

    # Read and check the initial marker
    initial_marker = struct.unpack("B", f.read(1))[0]
    if initial_marker != APPBIN_iCAMPO:
        print("Error: Configuration File Format Error at start marker.")
        return None, -1

    # Read and validate the data size field
    data_size = struct.unpack("I", f.read(4))[0]
    if data_size != DATA_SIZE:
        print("Error: Configuration File Format Error on data size.")
        return None, -1

    # Read the number of float32 elements
    n_data_file = struct.unpack("I", f.read(4))[0]
    if n_data_file > 0:
        data = struct.unpack(f"{n_data_file}f", f.read(n_data_file * DATA_SIZE))
    else:
        data = []

    # Read and check the final marker
    final_marker = struct.unpack("B", f.read(1))[0]
    if final_marker != APPBIN_fCAMPO:
        print("Error: Configuration File Format Error at end marker.")
        return None, -1

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]

    return data, error
