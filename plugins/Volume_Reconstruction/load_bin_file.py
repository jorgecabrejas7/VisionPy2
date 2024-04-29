import struct

from plugins.Volume_Reconstruction.field_loading import (
    read_bin_file_field_float32,
    read_bin_file_field_float64,
    read_bin_file_field_int16,
    read_bin_file_field_int32,
    read_bin_file_field_string,
    read_bin_file_field_uint16,
    read_bin_file_field_uint32,
    read_bin_file_field_uint64,
)

from plugins.Volume_Reconstruction.load_bin_v4 import load_bin_file_v4

load_bin_file_v5 = load_bin_file_v4


def load_bin_file(file_path):
    try:
        f = open(file_path, "rb")
    except IOError:
        print(f"Error opening the file {file_path}")
        return None, -1

    masc = struct.unpack("H", f.read(2))[0]
    if masc == 4660:
        pass
    elif masc == 13330:
        f.close()
        f = open(file_path, "rb", "big")
        masc = struct.unpack(">H", f.read(2))[0]
        if masc != 4660:
            print("Error in the control character")
            f.close()
            return None, -1
    else:
        print("Error in the control character")
        f.close()
        return None, -1

    version, error = read_bin_file_field_string(f)
    if error < 0:
        f.close()
        return None, error

    if "0000.0000.0000.0004" in version:
        print("Reading v4 file")
        data, error = load_bin_file_v4(f)
        if error < 0:
            f.close()
            return None, error
        data["version"] = version
    elif "0000.0000.0000.0005" in version:
        print("Reading v5 file")
        data, error = load_bin_file_v5(f)
        if error < 0:
            f.close()
            return None, error
        data["version"] = version
    else:
        print("Unsupported file version")
        f.close()
        return None, -1

    f.close()

    # # write data to a txt file
    # import json
    # with open("data.json", "w") as file:
    #     file.write(json.dumps(data, indent=4))

    print(f"File version: {data['version']}")
    print(f"File date: {data['date']}")
    print(f"Total Channels: {data['n_total_channels']}")
    print(f"Active Channels: {data['n_active_channels']}")
    print(f"Multiplexed Channels: {data['n_multiplexed_channels']}")
    print(f"stlib.dll v{data['dll_version']}")
    print(f"Boot Firmware v{data['firmware_boot_version']}")
    print(f"Ethernet Firmware v{data['firmware_ethernet_version']}")
    print(f"UCI Software v{data['firmware_uci_sw_version']}")
    print(f"UCI Hardware v{data['firmware_uci_hw_version']}")
    print(f"BASE v{data['firmware_base_version']}")
    print(f"MODULE v{data['firmware_module_version']}")

    return data, error
