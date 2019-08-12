import re


def get_question_template(question):
    """Generate question template from original question,
    including changing the number into number[x] format.
    从原始问题生成问题模板，包括修改数字为number[x]的形式。

    :param question:
    :return:
    """
    words = question.split(" ")
    start = 0
    index = 0
    number_variable_dict = {}
    variable_number_dict = {}
    for word in words:
        print(word)
        if is_numeric(word) and word in number_variable_dict:
            variable = "number_" + str(index)
            number_variable_dict[word] = variable
            variable_number_dict[variable] = word
            index += 1
            words[start] = variable
        start += 1
    question_template = " ".join(words)
    return question_template, number_variable_dict, variable_number_dict


def get_equations_template(equations, number_variable_dict):
    """Replace the numbers in equation to variables according to
    given number_variable_dict.
    根据number_variable_dict结构，按方程中的数字替换为对应的变量。

    :param equations:
    :param number_variable_dict:
    :return:
    """
    equations_template = []
    for equation in equations:
        words = equation.split(" ")
        start = 0
        for word in words:
            if is_numeric(word):
                if word not in number_variable_dict:
                    words[start] = number_variable_dict[word]
                else:
                    raise Exception("equation " + equation +
                                    " contains numbers which are not indexed")
            if word == 'x' or word == 'X':
                words[start] = "number_x"
            start += 1
        equation_template = " ".join(words)
        equations_template.append(equation_template)
    return equations_template


def recover_equations(equations, variable_number_dict):
    """
    根据variable_number_dict结构，按方程中的变量替换为对应的数字。

    :return:
    """
    equations_answer = []
    for equation in equations:
        words = equation.split(" ")
        start = 0
        for word in words:
            if variable_number_dict[word] is not None:
                words[start] = variable_number_dict[word]
            else:
                raise Exception("equation " + equation +
                                " contains variables which are not indexed")
            if word == 'number_x':
                words[start] = "x"
            start += 1
        equation_answer = " ".join(words)
        equations_answer.append(equation_answer)


def is_numeric(s):
    """ Determine whether the str matches a integer or a float number.
    """
    value = re.compile(r'^[-+]?\d*\.{0,1}\d+$')
    result = value.match(s)
    if result:
        return True
    else:
        return False