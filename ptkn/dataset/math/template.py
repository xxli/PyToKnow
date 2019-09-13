import re

from ptkn.dataset.math.mwp import MathWordProblem

number_prefix = "NUMBER"
unknown_suffix = "x"


def extract_question_number(question):
    """Generate question template from original question,
    including changing the number into number[x] format.
    从原始问题修改数字为number[x]的形式，并生成问题模板。

    :param question:
    :return:
    """
    words = question.split(" ")
    start = 0
    index = 0
    number_variable_dict = {}
    variable_number_dict = {}
    for word in words:
        # print(word)
        if _is_operand(word) and word not in number_variable_dict:
            variable = number_prefix + str(index)
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
    根据number_variable_dict结构，将方程中的数字替换为对应的变量。

    :param equations:
    :param number_variable_dict:
    :return:
    """
    equations_template = []
    for equation in equations:
        words = equation.split(" ")
        start = 0
        for word in words:
            if _is_operand(word):    # 1. if it is an operand. 如果是操作数。
                if word in number_variable_dict:
                    words[start] = number_variable_dict[word]
                else:
                    _found_variable= None
                    for number in number_variable_dict.keys():
                        if _number_match(word, number):
                            _found_variable = number_variable_dict[number]
                            break
                    if _found_variable is not None:
                        words[start] = _found_variable
                    else:
                        raise Exception("equation " + equation +
                                        " contains numbers which are not indexed")
            elif _is_operator(word):  # 2. If it is an operator. 如果是操作符。
                continue
            else:                     # 3. Otherwise, it is variable. 如果是其他，则默认为变量。
                words[start] = number_prefix + word
            start += 1
        equation_template = " ".join(words)
        equations_template.append(equation_template)
    return equations_template


def recover_equations(equations, variable_number_dict):
    """
    根据variable_number_dict，将方程中的变量替换为对应的数字。

    :return:
    """
    equations_answer = []
    for equation in equations:
        words = equation.split(" ")
        start = 0
        for word in words:
            if word in variable_number_dict:      # 1. If word is an variable, 如果词在变量-数值词典里。
                words[start] = variable_number_dict[word]

            elif word.startswith(number_prefix):  # 2. If word是待求解变量，
                words[start] = word.replace(number_prefix, "")
            start += 1
        equation_answer = " ".join(words)
        equations_answer.append(equation_answer)


def _is_operand(s):
    """
    determine whether s is an operand
    :param s:
    :return:
    """
    if _is_numeric(s) or s == "%" or _is_fraction(s):
        return True
    else:
        return False


def _is_numeric(s):
    """
    Determine whether the string matches a integer or a float number.
    """
    # integer_compiler = re.compile(r"[-+]?\d+")
    # float_complier = re.compile(r"[-+]?\d*\.{0,1}\d+")
    float_compiler_with_percent = re.compile(r"^[-+]?\d*\.{0,1}\d+%{0,1}$")
    result = float_compiler_with_percent.match(s)
    if result:
        return True
    else:
        return False


def _is_fraction(s):
    """
    Determine whether the string matches a fraction.
    :param s:
    :return:
    """
    fraction_compiler = re.compile(r"^\({0,1}\d+/\d+\){0,1}$")
    result = fraction_compiler.match(s)
    if result:
        return True
    else:
        return False


def _is_operator(s):
    """
    Determine whether the string is an operator.
    :param s:
    :return:
    """
    simple_operator_list = ["=", "+", "-", "*", "/"]
    if s in simple_operator_list:
        return True
    else:
        return False


def _number_match(number, word):
    return False


def _test_func():
    """
    test whether the functions in this python file is working correctly.
    :return:
    """
    _test_is_numeric()
    _test_is_fraction()


def _test_is_numeric():
    assert _is_numeric("14")
    assert _is_numeric("-.5")
    assert _is_numeric("14.20")
    assert _is_numeric("+14.20")
    assert _is_numeric("-14.20")
    assert _is_numeric("+14.20%")
    assert _is_numeric("14.2%")
    print("_test_is_numeric() no error.")


def _test_is_fraction():
    assert _is_fraction("1/5")
    assert _is_fraction("(2/17)")
    assert _is_fraction("(1/17)")
    print("_test_is_fraction() no error.")


def _test_equation_template():
    filename = r"D:\dataset\MaWPS\AllWithEquations.json"
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_mawps(filename)
        index_list, question_list, equations_list, solutions_list, \
        alignments_list, lqueryvars_list = alg.get_mawps_list()
        question_number = len(index_list)
        print("question number:", question_number)
        for i in range(question_number):
            index = index_list[i]
            question = question_list[i]
            equations = equations_list[i]
            solutions = solutions_list[i]
            alignments = alignments_list[i]
            lqueryvars = lqueryvars_list[i]
            print("question: ", question)
            question_template, number_variable_dict, variable_number_dict = extract_question_number(question)
            print("question_template: ", question_template)
            print("number_variable_dict: ", number_variable_dict)
            print("variable_number_dict: ", variable_number_dict)
            print("equations: ", equations)
            equations_template = get_equations_template(equations, number_variable_dict)
            print("equations: ", equations_template)
            new_equations = recover_equations(equations_template, variable_number_dict)
            print("new_equations: ", new_equations)
            if i == 0:
                break


if __name__ == "__main__":
    print()
    # _test_func()
    _test_equation_template()
