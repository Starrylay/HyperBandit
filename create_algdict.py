


def createSpecificAlgDict(specific, general, W, system_params, base_dict):
    # Define all of the required default arguments across all algorithms
    starter = createBaseAlgDict(specific, general, W, system_params)
    tmp = update_dict(specific, general)
    tmp2 = update_dict(tmp, base_dict)
    final_dict = update_dict(tmp2, starter)
    print(final_dict)
    return final_dict