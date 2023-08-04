def default_input(text, default):
    input_string = input(text)
    vartype = type(default[0])
    if input_string == "":
        return default
    else:
        return [vartype(value.strip()) for value in input_string.split(",")]


def candidate_input(name, candidates):
    print(f"Choose {name} from {candidates}")
    print(f"Example: {','.join(map(str, candidates))}")
    val = default_input("> ", candidates)
    if len(set(val) - set(candidates)) > 0:
        raise ValueError(f"{val} has a element which is not in {candidates}")
    return val
