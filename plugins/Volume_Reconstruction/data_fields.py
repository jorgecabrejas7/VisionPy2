from plugins.Volume_Reconstruction.field_loading import (
    read_bin_file_field_float32,
    read_bin_file_field_int16,
    read_bin_file_field_int32,
    read_bin_file_field_string,
    read_bin_file_field_uint32,
    read_bin_file_field_uint64,
)

FIELDS = [
    {
        "field_name": "date",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "dll_version",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "firmware_boot_version",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "firmware_ethernet_version",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "firmware_uci_sw_version",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "firmware_uci_hw_version",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "firmware_base_version",
        "read_function": read_bin_file_field_string,
        "message": None,
    },
    {
        "field_name": "firmware_module_version",
        "read_function": read_bin_file_field_string,
        "message": "Reading Version Modules... OK",
    },
    {
        "field_name": "n_total_channels",
        "read_function": read_bin_file_field_int32,
        "message": None,
    },
    {
        "field_name": "n_active_channels",
        "read_function": read_bin_file_field_int32,
        "message": None,
    },
    {
        "field_name": "n_multiplexed_channels",
        "read_function": read_bin_file_field_int32,
        "message": "Reading Channels Number... OK",
    },
    {
        "field_name": "trigger_lines_number",
        "read_function": read_bin_file_field_int32,
        "message": None,
    },
    {
        "field_name": "auxiliar_lines_number",
        "read_function": read_bin_file_field_int32,
        "message": None,
    },
    {
        "field_name": "trigger_scan_resolution",
        "read_function": read_bin_file_field_float32,
        "message": None,
    },
    {
        "field_name": "trigger_scan_length",
        "read_function": read_bin_file_field_float32,
        "message": None,
    },
    {
        "field_name": "auxiliar_scan_resolution",
        "read_function": read_bin_file_field_float32,
        "message": None,
    },
    {
        "field_name": "auxiliar_scan_length",
        "read_function": read_bin_file_field_float32,
        "message": "Reading Scan Lines Number... OK",
    },
    {
        "field_name": "trigger_source",
        "read_function": read_bin_file_field_int32,
        "message": "Reading Trigger Source... OK",
    },
    {
        "field_name": "prf_time_image",
        "read_function": read_bin_file_field_int32,
        "message": "Reading Scan PRF... OK",
    },
    {
        "field_name": "virtual_channels_number",
        "read_function": read_bin_file_field_int32,
        "message": "Reading Virtual Channels... OK",
    },
]


def get_vch_fields(i):
    return [
        {
            "field_name": "n_samples",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "n_ascan",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "start_range_mm",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "start_range_us",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "end_range_mm",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "end_range_us",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "sampling_frequency",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "reduction_factor",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "gain",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "prf_time_line",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "dynamic_range",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "x_entry_points",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "z_entry_points",
            "read_function": read_bin_file_field_float32,
            "message": None,
        },
        {
            "field_name": "angle_entry_points",
            "read_function": read_bin_file_field_float32,
            "message": f"[Virtual Channel {i}] Reading UT Config... OK",
        },
        {
            "field_name": "scan_raw",
            "read_function": read_bin_file_field_int16,
            "message": f"[Virtual Channel {i}] Reading RAW Image... OK",
        },
        {
            "field_name": "scan_dsp",
            "read_function": read_bin_file_field_int16,
            "message": f"[Virtual Channel {i}] Reading DSP Image... OK",
        },
        {
            "field_name": "scan_pci",
            "read_function": read_bin_file_field_int16,
            "message": f"[Virtual Channel {i}] Reading PCI Data... OK",
        },
        {
            "field_name": "scan_pci_img",
            "read_function": read_bin_file_field_int16,
            "message": f"[Virtual Channel {i}] Reading PCI Image... OK",
        },
        {
            "field_name": "sw_n_gates",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        # Same as above but with hw_n_gates
        {
            "field_name": "hw_n_gates",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "scan_counter",
            "read_function": read_bin_file_field_uint32,
            "message": f"[Virtual Channel {i}] Reading Images Counter... OK",
        },
        {
            "field_name": "n_encoders_values",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "n_encoders_edges_a",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "n_encoders_edges_b",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "n_encoders_glitches_a",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "n_encoders_glitches_b",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "n_encoders_sign_changes",
            "read_function": read_bin_file_field_int32,
            "message": None,
        },
        {
            "field_name": "encoder_offset",
            "read_function": read_bin_file_field_uint32,
            "message": None,
        },
        {
            "field_name": "scan_sw_timestamp",
            "read_function": read_bin_file_field_uint64,
            "message": None,
        },
        {
            "field_name": "scan_hw_timestamp",
            "read_function": read_bin_file_field_uint64,
            "message": None,
        },
        {
            "field_name": "scan_hw_free_memory",
            "read_function": read_bin_file_field_uint32,
            "message": None,
        },
    ]
