from plugins.Volume_Reconstruction.data_fields import FIELDS, get_vch_fields

from plugins.Volume_Reconstruction.field_loading import (
    read_bin_file_field_int16,
    read_bin_file_field_int32,
    read_bin_file_field_uint32,
)


def load_bin_file_v4(f):
    error = 0
    data = {}

    # Read basic information
    for field_info in FIELDS:
        field_name = field_info["field_name"]
        read_function = field_info["read_function"]
        message = field_info["message"]

        result, error = read_function(f)

        if error < 0:
            print(f"Error occurred while reading {field_name}.")
            return data, error
        if isinstance(result, tuple):
            data[field_name] = result[0]

        else:
            data[field_name] = result

        if message:
            print(message)

    # Read virtual channel data
    for i in range(data["virtual_channels_number"]):
        virtual_channel_data = {}

        for virtual_channel_info in get_vch_fields(i):
            field_name = virtual_channel_info["field_name"]
            read_function = virtual_channel_info["read_function"]
            message = virtual_channel_info["message"]

            if field_name == "sw_n_gates":
                n_gates, error = read_function(f)

                if error < 0:
                    print(f"Error occurred while reading n_gates for channel {i + 1}.")
                    return data, error

                if isinstance(n_gates, tuple):
                    n_gates = n_gates[0]

                for j in range(n_gates):
                    id_gate, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading id_gate for channel {i + 1}  of SW gate {j + 1}."
                        )
                        return data, error

                    peak_amplitude, error = read_bin_file_field_int16(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading peak_amplitude for channel {i + 1} of SW gate {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Peak Amplitude SW Gate {j}... OK"
                    )

                    peak_position, error = read_bin_file_field_int16(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading peak_position for channel {i + 1} of SW gate {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Peak Position SW Gate {j}... OK"
                    )

                    virtual_channel_data[f"sw_gate_{j + 1}"] = {
                        "id_gate": id_gate,
                        "peak_amplitude": peak_amplitude,
                        "peak_position": peak_position,
                    }
            elif field_name == "hw_n_gates":
                n_gates, error = read_function(f)

                if error < 0:
                    print(f"Error occurred while reading n_gates for channel {i + 1}.")
                    return data, error

                if isinstance(n_gates, tuple):
                    n_gates = n_gates[0]

                for j in range(n_gates):
                    id_gate, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading id_gate for channel {i + 1} of HW gate {j + 1}."
                        )
                        return data, error

                    peak_amplitude, error = read_bin_file_field_int16(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading peak_amplitude for channel {i + 1} of HW gate {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Peak Amplitude SW Gate {j}... OK"
                    )

                    peak_position, error = read_bin_file_field_int16(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading peak_position for channel {i + 1} of HW gate {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Peak Position SW Gate {j}... OK"
                    )

                    virtual_channel_data[f"hw_gate_{j + 1}"] = {
                        "id_gate": id_gate,
                        "peak_amplitude": peak_amplitude,
                        "peak_position": peak_position,
                    }

            elif field_name == "n_encoders_values":
                n_encoders, error = read_function(f)

                if error < 0:
                    print(
                        f"Error occurred while reading n_encoders (Values) for channel {i + 1}."
                    )
                    return data, error

                if isinstance(n_encoders, tuple):
                    n_encoders = n_encoders[0]

                for j in range(n_encoders):
                    data_int, error = read_bin_file_field_int32(
                        f
                    )  # No idea what this variable is for

                    if error < 0:
                        print(
                            f"Error occurred while reading data_int (Values) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    scan_encoder_value, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading scan_encoder_value for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    print(f"[Virtual Channel {i}] Reading Encoder {j} Value... OK")

                    if f"encoder_{j + 1}" not in virtual_channel_data:
                        virtual_channel_data[f"encoder_{j + 1}"] = {
                            "scan_encoder_value": scan_encoder_value
                        }

                    else:
                        virtual_channel_data[f"encoder_{j + 1}"][
                            "scan_encoder_value"
                        ] = scan_encoder_value

            elif field_name == "n_encoders_edges_a":
                n_encoders, error = read_function(f)
                if error < 0:
                    print(
                        f"Error occurred while reading n_encoders (Edges Channel A) for channel {i + 1}."
                    )
                    return data, error

                if isinstance(n_encoders, tuple):
                    n_encoders = n_encoders[0]

                for j in range(n_encoders):
                    data_int, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading data_int (Edges Channel A) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    scan_encoder_egdes_a, error = read_bin_file_field_uint32(f)
                    print("value of scan_encoder_egdes_a: ", scan_encoder_egdes_a)
                    if error < 0:
                        print(
                            f"Error occurred while reading scan_encoder_egdes (Edges Channel A) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Encoder {j} Edges Channel A... OK"
                    )

                    if f"encoder_{j + 1}" not in virtual_channel_data:
                        virtual_channel_data[f"encoder_{j + 1}"] = {
                            "scan_encoder_egdes_a": scan_encoder_egdes_a
                        }

                    else:
                        virtual_channel_data[f"encoder_{j + 1}"][
                            "scan_encoder_egdes_a"
                        ] = scan_encoder_egdes_a

            elif field_name == "n_encoders_edges_b":
                n_encoders, error = read_function(f)

                if error < 0:
                    print(
                        f"Error occurred while reading n_encoders (Edges Channel B) for channel {i + 1}."
                    )
                    return data, error

                if isinstance(n_encoders, tuple):
                    n_encoders = n_encoders[0]

                for j in range(n_encoders):
                    data_int, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading data_int (Edges Channel B) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    scan_encoder_egdes_b, error = read_bin_file_field_uint32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading scan_encoder_egdes (Edges Channel B) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Encoder {j} Edges Channel B... OK"
                    )

                    if f"encoder_{j + 1}" not in virtual_channel_data:
                        virtual_channel_data[f"encoder_{j + 1}"] = {
                            "scan_encoder_egdes_b": scan_encoder_egdes_b
                        }

                    else:
                        virtual_channel_data[f"encoder_{j + 1}"][
                            "scan_encoder_egdes_b"
                        ] = scan_encoder_egdes_b

            elif field_name == "n_encoders_glitches_a":
                n_encoders, error = read_function(f)

                if error < 0:
                    print(
                        f"Error occurred while reading n_encoders (Edges Glitches A) for channel {i + 1}."
                    )
                    return data, error

                if isinstance(n_encoders, tuple):
                    n_encoders = n_encoders[0]

                for j in range(n_encoders):
                    data_int, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading data_int (Edges Glitches A) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    scan_encoder_glitches_a, error = read_bin_file_field_uint32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading scan_encoder_glitches (Edges Glitches A) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Encoder {j} Glitches Channel A... OK"
                    )

                    if f"encoder_{j + 1}" not in virtual_channel_data:
                        virtual_channel_data[f"encoder_{j + 1}"] = {
                            "scan_encoder_glitches_a": scan_encoder_glitches_a
                        }

                    else:
                        virtual_channel_data[f"encoder_{j + 1}"][
                            "scan_encoder_glitches_a"
                        ] = scan_encoder_glitches_a

            elif field_name == "n_encoders_glitches_b":
                n_encoders, error = read_function(f)

                if error < 0:
                    print(
                        f"Error occurred while reading n_encoders (Edges Glitches B) for channel {i + 1}."
                    )
                    return data, error

                if isinstance(n_encoders, tuple):
                    n_encoders = n_encoders[0]

                for j in range(n_encoders):
                    data_int, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading data_int (Edges Glitches B) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    scan_encoder_glitches_b, error = read_bin_file_field_uint32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading scan_encoder_glitches (Edges Glitches B) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Encoder {j} Glitches Channel B... OK"
                    )

                    if f"encoder_{j + 1}" not in virtual_channel_data:
                        virtual_channel_data[f"encoder_{j + 1}"] = {
                            "scan_encoder_glitches_b": scan_encoder_glitches_b
                        }

                    else:
                        virtual_channel_data[f"encoder_{j + 1}"][
                            "scan_encoder_glitches_b"
                        ] = scan_encoder_glitches_b

            elif field_name == "n_encoders_sign_changes":
                n_encoders, error = read_function(f)

                if error < 0:
                    print(
                        f"Error occurred while reading n_encoders (Sign Changes) for channel {i + 1}."
                    )
                    return data, error

                if isinstance(n_encoders, tuple):
                    n_encoders = n_encoders[0]

                for j in range(n_encoders):
                    data_int, error = read_bin_file_field_int32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading data_int (Sign Changes) for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    sign_changes, error = read_bin_file_field_uint32(f)

                    if error < 0:
                        print(
                            f"Error occurred while reading sign_changes for channel {i + 1} of encoder {j + 1}."
                        )
                        return data, error

                    print(
                        f"[Virtual Channel {i}] Reading Encoder {j} Sign Changes... OK"
                    )

                    if f"encoder_{j + 1}" not in virtual_channel_data:
                        virtual_channel_data[f"encoder_{j + 1}"] = {
                            "sign_changes": sign_changes
                        }

                    else:
                        virtual_channel_data[f"encoder_{j + 1}"]["sign_changes"] = (
                            sign_changes
                        )

            else:
                virtual_channel_data[field_name], error = read_function(f)

                if error < 0:
                    print(
                        f"Error occurred while reading {field_name} for channel {i + 1}."
                    )
                    return data, error

                if message:
                    print(message)

        # Add code here to read virtual channel data
        # You can use a similar loop structure as the one above for basic information

        data[f"virtual_channel_{i + 1}"] = virtual_channel_data

    return data, error
