import sys


def cross_validation(k, *data_list):
    """ [item]-cross validation on one list
    arg:
        data_lists: list of examples

    return: n pairs of training and test dataset, each dataset in every pair \
        is a list of examples. The [[train-1, test-1], ..., [train-k, test-k]]
    """

    assert isinstance(k, int), "k must be integer."
    assert k > 1, "k must be larger than 1."
    assert k <= 100, "k is larger than 100. Please check."

    # data_size denotes how many kinds of data
    data_size = len(data_list)
    assert data_size >= 1, "no data input"
    print("the size of data is ", data_size)

    # data_length denotes the length of each data
    data_length = len(data_list[0])
    for i in range(1, data_size):
        assert len(data_list[i]) == data_length, "The input data have different lengths."
    print("the length for each data is ", data_length)

    cross_size = int(data_length/k)

    # output_data_list format: [A [B [C [D] ] ] ]
    # A: contain k elements (k-cross validation)
    # B: contain 2 elements (training and testing)
    # C: contain data_size elements (how many different types of data)
    # D: contain data_length*(k-1)/k, data_length/k elements (for different B)
    output_data_list = []
    for a in range(k):
        output_data_list.append([])
        for b in range(2):
            output_data_list[a].append([])
            for c in range(data_size):
                output_data_list[a][b].append([])
    # print(output_data_list)

    for c in range(data_size):
        for d in range(data_length):
            element = data_list[c][d]
            # current element belong to
            the_k = d // cross_size
            for a in range(k):
                if the_k == a:
                    output_data_list[a][1][c].append(element)
                else:
                    output_data_list[a][0][c].append(element)

    for a in range(k):
        for b in range(2):
            if data_size == 1:
                output_data_list[a][b] = output_data_list[a][b][0]
            else:
                output_data_list[a][b] = tuple(output_data_list[a][b])

    return output_data_list


if __name__ == "__main__":
    list_one = [1, 2, 3, 4, 5]
    list_result = cross_validation(5, list_one)
    print(list_result)

    list_two = ["a", "b", "c", "d", "e"]

    list_result = cross_validation(5, list_one, list_two)
    training_1, training_2 = list_result[0][0]
    test_1, test_2  = list_result[0][1]
    print(training_1, training_2)
    print(test_1, test_2)
