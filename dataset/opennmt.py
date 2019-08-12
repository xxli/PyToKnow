
def output_opennmt(input_list, filename):
    """output mwp list to files

    :param input_list:
    :param filename:
    :return:
    """
    assert isinstance(input_list, list), "input_list is not a list."
    train_file = open(filename, 'w')
    for i in range(len(input_list)):
        line = input_list[i]
        train_file.write(line + "\n")
    train_file.close()