import json
import re


class AlgQuestion:
    """
    Class for math word problem, including AI2-395, Alg-514, and SingleEQ dataset.
    AI2数学题的类。
    """

    def __init__(self):
        self.index = 0
        # Question contains one or more sentences.
        self.question = ""
        # equation like "X = 70 - 27".
        self.equations = []
        # the solution/answer of question.
        self.solutions = []
        #


class ILQuestion(AlgQuestion):
    """
    Class for math word problem, including IL-562 and Commoncore-600 dataset.
    """

    def __init__(self):
        super().__init__()
        self.alignments = []


class MAWPSQuestion(AlgQuestion):
    """
    Class for math word problem, including MAWPS dataset.
    """

    def __init__(self):
        super().__init__()
        self.alignments = []
        self.lqueryvars = []


class ArithQuestion(ILQuestion):
    """
    Class for math word problem, including AllArith dataset.
    """

    def __init__(self):
        super().__init__()
        self.quants = []
        self.rates = []


class DrawQuestion(AlgQuestion):
    """
    Class for math word problem, including Draw-1K
    """

    def __init__(self):
        super().__init__()
        self.template = []
        self.alignment = []
        self.equiv = []


class DolphinQuestion():
    """
    Class for math word problem, including Dolphin1978, Dolphin18k
    """

    def __int__(self):
        self.id = ""
        # self.index = ""
        self.text = ""
        # self.sources = ""
        self.equations = []
        self.ans = ""
        self.ans_simple = []


class MathQuestion:

    def __init__(self):
        self.id = ""
        self.original_text = ""
        self.segmented_text = ""
        self.equation = ""
        self.ans = ""


class AQUAQuestion:

    def __init__(self):
        self.question = ""
        self.options = []
        self.rationale = ""
        self.correct = ""


class Alignment:

    def __init__(self):
        self.coeff = ""
        self.sentence_id = 0
        self.value = 0.0
        self.token_id = 0.0

    def assign(self, parsed_json=None):
        if parsed_json is None:
            parsed_json = {}
        self.coeff = parsed_json["coeff"]
        self.sentence_id = parsed_json["SentenceId"]
        self.value = parsed_json["Value"]
        self.token_id = parsed_json["TokenId"]


class MathWordProblem:

    def __init__(self):
        self.mwp_list = list()

    def read_alg(self, filename):
        """
        read AI2 dataset

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = AlgQuestion()
                mwp.index = question_json['iIndex']
                mwp.question = question_json['sQuestion']
                mwp.equations = question_json['lEquations']
                mwp.solutions = question_json['lSolutions']
                self.mwp_list.append(mwp)

    def read_il(self, filename):
        """
        read IL-600 dataset

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = ILQuestion()
                mwp.index = question_json['iIndex']
                mwp.question = question_json['sQuestion']
                mwp.equations = question_json['lEquations']
                mwp.solutions = question_json['lSolutions']
                mwp.alignments = question_json['lAlignments']
                self.mwp_list.append(mwp)

    def read_mawps(self, filename):
        """
        read IL-600 dataset

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = MAWPSQuestion()
                mwp.index = question_json['iIndex']
                mwp.question = question_json['sQuestion']
                mwp.equations = question_json['lEquations']
                mwp.solutions = question_json['lSolutions']
                if 'lAlignments' in question_json:
                    mwp.alignments = question_json['lAlignments']
                if 'lQueryVars' in question_json:
                    mwp.lqueryvars = question_json['lQueryVars']

                self.mwp_list.append(mwp)

    def read_arith(self, filename):
        """
        read AllArith dataset

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = ArithQuestion()
                mwp.index = question_json['iIndex']
                mwp.question = question_json['sQuestion']
                mwp.equations = question_json['lEquations']
                mwp.solutions = question_json['lSolutions']
                mwp.alignments = question_json['lAlignments']
                mwp.quants = question_json['quants']
                mwp.rates = question_json['rates']
                self.mwp_list.append(mwp)

    def read_draw(self, filename):
        """
        read Draw-1K data set

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = DrawQuestion()
                mwp.index = question_json['iIndex']
                mwp.question = question_json['sQuestion']
                mwp.equations = question_json['lEquations']
                mwp.solutions = question_json['lSolutions']
                mwp.template = question_json["Template"]
                mwp.alignment = question_json['Alignment']
                mwp.equiv = question_json['Equiv']
                self.mwp_list.append(mwp)

    def read_dolphin(self, filename):
        """
        read Dolphin data set

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = DolphinQuestion()
                mwp.id = question_json['id']
                # mwp.index = question_json['index']
                mwp.text = question_json['text']
                # mwp.sources = question_json['sources']
                mwp.equations = question_json['equations']
                mwp.ans = question_json['ans']
                mwp.ans_simple = question_json['ans_simple'] if 'ans_simple' in question_json else []
                self.mwp_list.append(mwp)

    def read_math(self, filename):
        """
        read Math23k data set

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            question_lines = question_lines.replace("}", "},").strip()
            question_lines = "[ " + question_lines[:-1] + " ]"
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = MathQuestion()
                mwp.id = question_json['id']
                mwp.original_text = question_json['original_text']
                mwp.segmented_text = question_json['segmented_text']
                mwp.equation = question_json['equation']
                mwp.ans = question_json['ans']
                self.mwp_list.append(mwp)

    def read_aqua(self, filename):
        """
        read AQUA data set

        return: mwp list
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            question_lines = file.read()
            question_lines = question_lines.replace("}", "},").strip()
            question_lines = "[ " + question_lines[:-1] + " ]"
            questions_json = json.loads(question_lines)
            for question_json in questions_json:
                mwp = AQUAQuestion()
                mwp.question = question_json['question']
                mwp.options = question_json['options']
                mwp.rationale = question_json['rationale']
                mwp.correct = question_json['correct']
                self.mwp_list.append(mwp)

    def get_alg_list(self):
        """

        :return: index_list, question_list, equation_list, solution_list
        """
        index_list = []
        question_list = []
        equations_list = []
        solutions_list = []
        for mwp in self.mwp_list:
            index_list.append(mwp.index)
            question_list.append(mwp.question)
            equations_list.append(mwp.equations)
            solutions_list.append(mwp.solutions)
        return index_list, question_list, equations_list, solutions_list

    def get_il_list(self):
        """

        :return: index_list, question_list, equation_list, solution_list
        """
        index_list = []
        question_list = []
        equations_list = []
        solutions_list = []
        alignments_list = []
        for mwp in self.mwp_list:
            index_list.append(mwp.index)
            question_list.append(mwp.question)
            equations_list.append(mwp.equations)
            solutions_list.append(mwp.solutions)
            alignments_list.append(mwp.alignments)
        return index_list, question_list, equations_list, solutions_list, \
               alignments_list

    def get_mawps_list(self):
        """

        :return: index_list, question_list, equation_list, solution_list
        """
        index_list = []
        question_list = []
        equations_list = []
        solutions_list = []
        alignments_list = []
        lqueryvars_list = []
        for mwp in self.mwp_list:
            index_list.append(mwp.index)
            question_list.append(mwp.question)
            equations_list.append(mwp.equations)
            solutions_list.append(mwp.solutions)
            alignments_list.append(mwp.alignments)
            lqueryvars_list.append(mwp.lqueryvars)
        return index_list, question_list, equations_list, solutions_list, \
               alignments_list, lqueryvars_list

    def get_arith_list(self):
        """

        :return: index_list, question_list, equation_list, solution_list
        """
        index_list = []
        question_list = []
        equations_list = []
        solutions_list = []
        alignments_list = []
        quants_list = []
        rates_list = []
        for mwp in self.mwp_list:
            index_list.append(mwp.index)
            question_list.append(mwp.question)
            equations_list.append(mwp.equations)
            solutions_list.append(mwp.solutions)
            alignments_list.append(mwp.alignments)
            quants_list.append(mwp.quants)
            rates_list.append(mwp.rates)
        return index_list, question_list, equations_list, solutions_list, \
               alignments_list, quants_list, rates_list

    def get_draw_list(self):
        """

        :return: index_list, question_list, equation_list, solution_list
        """
        index_list = []
        question_list = []
        equations_list = []
        solutions_list = []
        alignment_list = []
        template_list = []
        equiv_list = []
        for mwp in self.mwp_list:
            index_list.append(mwp.index)
            question_list.append(mwp.question)
            equations_list.append(mwp.equations)
            solutions_list.append(mwp.solutions)
            alignment_list.append(mwp.alignment)
            template_list.append(mwp.template)
            equiv_list.append(mwp.equiv)

        return index_list, question_list, equations_list, solutions_list, \
               alignment_list, template_list, equiv_list

    def get_dolphin_list(self):
        """

        :return: index_list, question_list, equation_list, solution_list
        """
        id_list = []
        # index_list = []
        text_list = []
        # sources_list = []
        equations_list = []
        ans_list = []
        ans_simple_list = []
        for mwp in self.mwp_list:
            id_list.append(mwp.id)
            # index_list.append(mwp.index)
            text_list.append(mwp.text)
            # sources_list.append(mwp.sources)
            equations_list.append(mwp.equations)
            ans_list.append(mwp.ans)
            ans_simple_list.append(mwp.ans_simple)

        return id_list, text_list, \
               equations_list, ans_list, ans_simple_list

    def get_math_list(self):
        """

        :return:
        """
        id_list = []
        original_text_list = []
        segmented_text_list = []
        equation_list = []
        ans_list = []
        for mwp in self.mwp_list:
            id_list.append(mwp.id)
            original_text_list.append(mwp.original_text)
            segmented_text_list.append(mwp.segmented_text)
            equation_list.append(mwp.equation)
            ans_list.append(mwp.ans)

        return id_list, original_text_list, segmented_text_list, \
               equation_list, ans_list

    def get_aqua_list(self):
        """

        :return:
        """
        question_list = []
        options_list = []
        rationale_list = []
        correct_list = []
        for mwp in self.mwp_list:
            question_list.append(mwp.question)
            options_list.append(mwp.options)
            rationale_list.append(mwp.rationale)
            correct_list.append(mwp.correct)
        return question_list, options_list, rationale_list, correct_list


def test_alg(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_alg(filename)
        index_list, question_list, equations_list, solutions_list = \
            alg.get_alg_list()
        print("question number:", len(index_list))
        for i in range(len(index_list)):
            index = index_list[i]
            question = question_list[i]
            equations = equations_list[i]
            solutions = solutions_list[i]
            if i == 0 or i == len(index_list) - 1:
                print(index)
                print(question)
                print(equations)
                print(len(equations))
                print(solutions)
                print(len(solutions))


def test_il(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_il(filename)
        index_list, question_list, equations_list, solutions_list, \
        alignments_list = alg.get_il_list()
        print("question number:", len(index_list))
        for i in range(len(index_list)):
            index = index_list[i]
            question = question_list[i]
            equations = equations_list[i]
            solutions = solutions_list[i]
            alignments = alignments_list[i]
            if i == 0 or i == len(index_list) - 1:
                print(index)
                print(question)
                print(equations)
                print(len(equations))
                print(solutions)
                print(len(solutions))
                print(alignments)
                print(len(alignments))


def test_mawps(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_mawps(filename)
        index_list, question_list, equations_list, solutions_list, \
        alignments_list, lqueryvars_list = alg.get_mawps_list()
        print("question number:", len(index_list))
        for i in range(len(index_list)):
            index = index_list[i]
            question = question_list[i]
            equations = equations_list[i]
            solutions = solutions_list[i]
            alignments = alignments_list[i]
            lqueryvars = lqueryvars_list[i]
            if i in [0,  395, 2245, 2845, 3353]:
                print(index)
                print(question)
                print(equations)
                print(len(equations))
                print(solutions)
                print(len(solutions))
                print(alignments)
                print(len(alignments))
                print(lqueryvars)
                print(len(lqueryvars))


def test_arith(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_arith(filename)
        index_list, question_list, equations_list, solutions_list, \
            alignments_list, quants_list, rates_list = alg.get_arith_list()
        print("question number:", len(index_list))
        for i in range(len(index_list)):
            index = index_list[i]
            question = question_list[i]
            equations = equations_list[i]
            solutions = solutions_list[i]
            alignments = alignments_list[i]
            quants = quants_list[i]
            rates = rates_list[i]
            if i == 0 or i == len(index_list) - 1:
                print(index)
                print(question)
                print(equations)
                print(len(equations))
                print(solutions)
                print(len(solutions))
                print(alignments)
                print(len(alignments))
                print(quants)
                print(len(quants))
                print(rates)
                print(len(rates))


def test_draw(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_draw(filename)
        index_list, question_list, equations_list, solutions_list, \
        alignment_list, template_list, equiv_list = alg.get_draw_list()
        print("question number:", len(index_list))
        for i in range(len(index_list)):
            index = index_list[i]
            question = question_list[i]
            equations = equations_list[i]
            solutions = solutions_list[i]
            template = template_list[i]
            alignment = alignment_list[i]
            equiv = equiv_list[i]
            if i == 0 or i == len(index_list) - 1:
                print(index)
                print(question)
                print(equations)
                print(len(equations))
                print(solutions)
                print(len(solutions))
                print(template)
                print(len(template))
                print(alignment)
                print(len(alignment))
                print(equiv)
                print(len(equiv))
                if len(equiv) > 0:
                    print(equiv[0])


def test_dolphin(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_dolphin(filename)
        id_list, text_list, \
            equations_list, ans_list, ans_simple_list = alg.get_dolphin_list()
        print("question number:", len(id_list))
        for i in range(len(id_list)):
            id = id_list[i]
            # index = index_list[i]
            text = text_list[i]
            # sources = sources_list[i]
            equations = equations_list[i]
            ans = ans_list[i]
            ans_simple = ans_simple_list[i]
            if i == 0 or i == len(id_list) - 1:
                print(id)
                # print(index)
                print(text)
                # print(sources)
                print(equations)
                print(len(equations))
                print(ans)
                print(ans_simple)
                print(len(ans_simple))


def test_math(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_math(filename)
        id_list, original_text_list, segmented_text_list, \
            equation_list, ans_list = alg.get_math_list()
        print("question number:", len(id_list))
        for i in range(len(id_list)):
            id = id_list[i]
            original_text = original_text_list[i]
            segmented_text = segmented_text_list[i]
            equation = equation_list[i]
            ans = ans_list[i]
            if i == 0 or i == len(id_list) - 1:
                print(id)
                print(original_text)
                print(segmented_text)
                print(equation)
                print(ans)


def test_aqua(filename):
    if filename is not None and len(filename) > 0:
        alg = MathWordProblem()
        alg.read_aqua(filename)
        question_list, options_list, rationale_list, correct_list \
            = alg.get_aqua_list()
        print("question number:", len(question_list))
        for i in range(len(question_list)):
            question = question_list[i]
            options = options_list[i]
            rationale = rationale_list[i]
            correct = correct_list[i]
            if i == 0 or i == len(question_list) - 1:
                print(question)
                print(options)
                print(len(options))
                print(rationale)
                print(correct)


def read_test():
    # test_alg(r"D:\dataset\AI2-395\AddSub.json") # AI2
    # test_alg(r"D:\dataset\MaWPS\AddSub.json") # AI2
    # test_alg(r"D:\dataset\Alg-514\questions.json") # Alg-514
    # test_alg(r"D:\dataset\MaWPS\Kushman.json") # Alg-514
    # test_alg(r"D:\dataset\SingleEQ\questions.json")  # SingleEQ
    # test_alg(r"D:\dataset\MaWPS\SingleEQ.json") # SingleEQ
    # test_il(r"D:\dataset\IL-562\questions.json")  # IL-562
    # test_il(r"D:\dataset\MaWPS\SingleOp.json") # IL-562
    # test_il(r"D:\dataset\Commoncore-600\questions.json")  # Commoncore-600
    # test_il(r"D:\dataset\MaWPS\MultiArith.json")  # Commoncore-600
    test_mawps(r"D:\dataset\MaWPS\AllWithEquations.json")  # MAWPS
    # test_arith(r"D:\dataset\AllArith\questions.json")  # AllArith
    # test_draw(r"D:\dataset\DRAW-1K\draw.json")  # Draw-1K
    # test_dolphin(r"D:\dataset\dolphin\dolphin-number_word_std\number_word_std.dev.json")  # Dolphin1878
    # test_dolphin(r"D:\dataset\dolphin\dolphin18k\dev_cleaned.json")  # Dolphin18k
    # test_math(r"D:\dataset\Math23K\Math23k_test.json")  # Math23k
    # test_math(r"D:\dataset\Math23K\Math23k_train.json")  # Math23k
    # test_aqua(r"D:\dataset\AQUA-RAT\dev.json")  # AQUA-RAT


if __name__ == "__main__":
    read_test()
    # generateDataset(sys.argv[1], sys.argv[2], sys.argv[3])
