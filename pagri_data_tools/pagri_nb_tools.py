from nbformat import v4 as nb_v4, reads as nb_reads, read as nb_read, write as nb_write
import json
import re
from pyaspeller import YandexSpeller
from IPython.display import display_html
import IPython
from tqdm.auto import tqdm
# from nltk.tokenize import word_tokenize


def add_links_and_numbers_to_headings(notebook_path: str, mode: str = 'draft', link_type: str = "name", start_level: int = 2, add_link_to_content = False):
    """
    Добавляет ссылки к заголовкам в ноутбуке.

    Args:
        notebook_path (str): Путь к ноутбуку.
        link_type (str, optional): Тип ссылки. Может быть "name" или "id". Defaults to "name".
        start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        mode (str): Режим работы функции. Либо 'draft', в этом случае создаться копия файла, либо 'final' в этом случае изменения будут сделаны в исходном файле. Default to 'draft'
    Returns:
        None

    Example:
        - Вариант, который работает почти везде
            в оглавлении пишем
            <a href="#Глава-1">Ссылка на главу 1</a>
            в названии главы пишем
            # 1 Введение <a name="Глава-1"></a>
        - Вариант, который не везде работает
            в оглавлении пишем
            [Ссылка на главу 1](#name-id)
            в названии главы пишем
            <a class="anchor" id="Название-главы"></a>
            ### Название главы
        - Упрощенный вариант, работает только в jupyter notebook
            в оглавлении пишем
            [Ссылка на главу 1](#Глава-1)
            в названии главы пишем
            # 1 Введение
    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['name', 'id']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'name' or 'id'.")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    # headers = []
    # Создаем список счетчиков для каждого уровня заголовка
    counters = [0] * 100
    is_previous_cell_header = False
    for cell in nb_json["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"]
            new_source = []
            for line in source:
                level = line.count("#")
                if level >= start_level and not line.strip().endswith("</a>") and not line.strip().endswith("<skip>"):
                    # Уменьшаем на 1, чтобы название отчета не нумеровалось
                    level = level - 1
                    title = line.strip().lstrip("#").lstrip()
                    # Обновляем счетчики
                    counters[level - 1] += 1
                    for i in range(level, len(counters)):
                        counters[i] = 0
                    number_parts = [str(counter)
                                    for counter in counters[:level]]
                    chapter_number = ".".join(number_parts)
                    # для глав с одной цифрой добавляем точку, чтобы было 1. Название главы, для остальных уровней в конкце не будет точки
                    if level + 1 == start_level:
                        chapter_number += '.'
                    # Формируем пронумерованный заголовок
                    title_text = f"{(level + 1) * '#'} {chapter_number} {title.strip()}"
                    text_for_ref = f'{chapter_number}-{title.strip()}'
                    text_for_ref = regex_for_sub.sub('-', text_for_ref)
                    if link_type == "name":
                        header = f"{title_text}<a name='{text_for_ref}'></a>"
                    elif link_type == "id":
                        header = f"{title_text}<a class='anchor' id='{text_for_ref}'></a>"
                    else:
                        raise ValueError(
                            "Неправильный тип ссылки. Должно быть 'name' или 'id'.")
                    if add_link_to_content and not is_previous_cell_header:
                        header += '\n<a href="#ref-to-toc">вернуться к оглавлению</a>'
                    new_source.append(header)
                    is_previous_cell_header = True
                else:
                    is_previous_cell_header = False
                    new_source.append(line)
            cell["source"] = new_source
        else:
            is_previous_cell_header = False

    # Convert nb_json to nbformat object
    nb = nb_reads(json.dumps(nb_json), as_version=4)

    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Numbers of headers and links added to {output_filename}")


def generate_toc(notebook_path: str, mode: str = 'draft', indent_char: str = "&emsp;", link_type: str = "html", start_level: int = 2):
    """
    Генерирует оглавление для ноутбука.

    Args:
        notebook_path (str): Путь к ноутбуку.
        indent_char (str, optional): Символ для отступа. Defaults to "&emsp;".
        link_type (str, optional): Тип ссылки. Может быть "markdown" или "html". Defaults to "html".
        start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        mode (str): Режим работы функции. Либо 'draft', в этом случае создаться копия файла, либо 'final' в этом случае изменения будут сделаны в исходном файле. Default to 'draft'

    Returns:
        None

    Example:
        Для link_type="markdown":
            &emsp;[Глава-1](#Глава-1)<br>
            &emsp;&emsp;[Подглава-1.1](#Подглава-1.1)<br>
        Для link_type="html":
            &emsp;<a href="Глава-1">Глава 1</a><br>
            &emsp;&emsp;<a href="#Подглава-1.1">Подглава 1.1</a><br>

        - Вариант, который работает почти везде
            в оглавлении пишем
            <a href="#лава-1">Ссылка на главу 1</a>
            в названии главы пишем
            # Глава 1 <a name="Глава-1"></a>
        - Вариант, который не везде работает
            в оглавлении пишем
            [Ссылка на главу 1](#name-id)
            в названии главы пишем
            <a class="anchor" id="Название-главы"></a>
            ### Название главы
        - Упрощенный вариант, работает только в jupyter notebook
            в оглавлении пишем
            [Ссылка на главу 1](#Глава-1)
            в названии главы пишем
            # Глава 1
    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['markdown', 'html']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'markdown' or 'html'.")
    if mode == 'draft':
        notebook_path_splited = notebook_path.split('.')
        notebook_path_splited[-2] += '_temp'
        notebook_path = '.'.join(notebook_path_splited)

    def is_markdown(it): return "markdown" == it["cell_type"]
    def is_title(it): return it.strip().startswith(
        "#") and it.strip().lstrip("#").lstrip()
    toc = ['**Оглавление**<a name="ref-to-toc"></a>\n\n',]
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    for cell in filter(is_markdown, nb_json["cells"]):
        for line in filter(is_title, cell["source"]):
            level = line.count("#")
            if level < start_level or line.endswith('<skip>'):
                continue
            line = line.strip()
            indent = indent_char * (level * 2 - 3)
            title_for_ref = re.findall(
                r'<a name=[\"\']+(.*)[\"\']+></a>', line)
            if not title_for_ref:
                raise ValueError(
                    f'В строке "{line}" нет ссылки для создания оглавления')
            title_for_ref = title_for_ref[0]
            title = re.sub(r'<a.*</a>', '', line).lstrip("#").lstrip()
            if link_type == "markdown":
                toc_line = f"{indent}[{title}](#{title_for_ref})  \n"
            elif link_type == "html":
                toc_line = f"{indent}<a href='#{title_for_ref}'>{title}</a>  \n"
            else:
                raise ValueError(
                    "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
            toc.append(toc_line)
    toc_cell = nb_v4.new_markdown_cell([''.join(toc)])
    nb_json['cells'].insert(0, toc_cell)
    # display(nb_json)
    # Convert nb_json to nbformat object
    nb = nb_reads(json.dumps(nb_json), as_version=4)

    # Save the nbformat object to the file
    output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Table of content added to {output_filename}")


def make_headers_link_and_toc(notebook_path: str, mode: str = 'draft', start_level: int = 2, link_type_header: str = "name", indent_char: str = "&emsp;", link_type_toc: str = "html", is_make_headers_link: bool = True, is_make_toc: bool = False):
    ''' 
    Функция добавляет ссылки в название headers и создает содеражние

    Args:
        - notebook_path (str): Путь к ноутбуку.
        - mode (str): Режим работы функции. Либо 'draft', в этом случае создаться копия файла, либо 'final' в этом случае изменения будут сделаны в исходном файле. Default to 'draft'
        - link_type_header (str, optional): Тип ссылки в заголовке. Может быть "name" или "id". Defaults to "name".
        - start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        - indent_char (str, optional): Символ для отступа в оглавлении. Defaults to "&emsp;".
        - link_type_toc (str, optional): Тип ссылки в оглавлении на заголовок. Может быть "markdown" или "html". Defaults to "html".
        - start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        - is_make_headers_link (book): Делать ссылки в заголовках. Defaults to 'True'
        - is_make_toc (book): Делать оглавление. Defaults to 'True'
    Returns:
        None    
    Example:
        Для link_type="markdown":
            &emsp;[Глава-1](#Глава-1)<br>
            &emsp;&emsp;[Подглава-1.1](#Подглава-1.1)<br>
        Для link_type="html":
            &emsp;<a href="Глава-1">Глава 1</a><br>
            &emsp;&emsp;<a href="#Подглава-1.1">Подглава 1.1</a><br>

        - Вариант, который работает почти везде
            в оглавлении пишем
            <a href="#лава-1">Ссылка на главу 1</a>
            в названии главы пишем
            # Глава 1 <a name="Глава-1"></a>
        - Вариант, который не везде работает
            в оглавлении пишем
            [Ссылка на главу 1](#name-id)
            в названии главы пишем
            <a class="anchor" id="Название-главы"></a>
            ### Название главы
        - Упрощенный вариант, работает только в jupyter notebook
            в оглавлении пишем
            [Ссылка на главу 1](#Глава-1)
            в названии главы пишем
            # Глава 1
    '''
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type_header not in ['name', 'id']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'name' or 'id'.")
    if link_type_toc not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")
    if is_make_headers_link:
        add_links_and_numbers_to_headings(
            notebook_path, mode=mode, link_type=link_type_header, start_level=start_level)
    if is_make_toc:
        generate_toc(notebook_path, mode=mode, link_type=link_type_toc,
                     start_level=start_level, indent_char=indent_char)


def add_conclusions_and_anomalies_old(notebook_path: str, mode: str = 'draft', link_type: str = 'html', order: dict = None):
    """
    This function adds conclusions and anomalies sections to a Jupyter notebook.

    Args:
        notebook_path (str): The path to the Jupyter notebook file.
        mode (str): The mode of the output file, either 'draft' or 'final'. Defaults to 'draft'.
        link_type (str): The type of link to use, either 'html' or 'markdown'. Defaults to 'html'.
        order (dict): dict of lists with ordered  conclusions and anomalie (key: conclusions and anomalies)

    Examples:
        order = dict(
                    conclusions =[ 'Женщины чаще возвращают кредит, чем мужчины.']
                    , anomalies = ['В датафрейме есть строки дубликаты. 54 строки. Меньше 1 % от всего датафрейма.  ']
                )
    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    conclusions = []
    anomalies = []

    for cell in nb_json["cells"]:
        cell_has_ref_to_toc = False
        source = cell["source"]
        new_source = []
        for line in source:
            if line.strip().startswith("_conclusion_") or line.strip().startswith("_anomalies_"):
                cell["cell_type"] = "markdown"
                if line.strip().startswith("_conclusion_"):
                    conclusion_or_anomaly = line.strip().replace("_conclusion_ ", '')
                if line.strip().startswith("_anomalies_"):
                    conclusion_or_anomaly = line.strip().replace("_anomalies_ ", '')
                conclusion_or_anomaly_for_ref = regex_for_sub.sub(
                    '-', conclusion_or_anomaly)
                if link_type == "html":
                    toc_conclusion_or_anomaly = f"- <a href='#{conclusion_or_anomaly_for_ref}'>{conclusion_or_anomaly}</a>  \n"
                elif link_type == "markdown":
                    toc_conclusion_or_anomaly = f"[- {conclusion_or_anomaly}](#{conclusion_or_anomaly_for_ref})  \n"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
                if link_type == "html":
                    conclusion_or_anomaly_for_ref = f"<a name='{conclusion_or_anomaly_for_ref}'></a>"
                elif link_type == "markdown":
                    conclusion_or_anomaly_for_ref = f"<a class='anchor' id='{conclusion_or_anomaly_for_ref}'></a>"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'name' или 'id'.")

                if line.strip().startswith("_conclusion_"):
                    conclusions.append(
                        (toc_conclusion_or_anomaly, conclusion_or_anomaly))
                    if not cell_has_ref_to_toc:
                        conclusion_or_anomaly_for_ref += '\n<a href="#ref-to-conclusions">Вернуться к выводам</a>'
                if line.strip().startswith("_anomalies_"):
                    anomalies.append(
                        (toc_conclusion_or_anomaly, conclusion_or_anomaly))
                    if not cell_has_ref_to_toc:
                        conclusion_or_anomaly_for_ref += '\n<a href="#ref-to-anomalies">Вернуться к выводам</a>'
                new_source.append(conclusion_or_anomaly_for_ref)
                cell_has_ref_to_toc = True
            else:
                new_source.append(line)
        cell["source"] = new_source
    # Отсортируем  выводы и аномалии как в переданном словаре
    if order:
        ordered_conclusions = order['conclusions']
        ordered_anomalies = order['anomalies']
        index_map = {x.strip(): i for i, x in enumerate(ordered_conclusions)}
        conclusions = sorted(
            conclusions, key=lambda x: index_map[x[1].strip()])
        index_map = {x.strip(): i for i, x in enumerate(ordered_anomalies)}
        anomalies = sorted(anomalies, key=lambda x: index_map[x[1].strip()])
    conclusions = [conclusion[0] for conclusion in conclusions]
    anomalies = [anomalie[0] for anomalie in anomalies]
    conclusions = [
        '**Главные выводы:**<a name="ref-to-conclusions"></a>  \n'] + conclusions
    anomalies = [
        '**Аномалии и особенности в данных:**<a name="ref-to-anomalies"></a>  \n'] + anomalies
    conclusions_cell = nb_v4.new_markdown_cell([''.join(conclusions)])
    anomalies_cell = nb_v4.new_markdown_cell([''.join(anomalies)])
    nb_json['cells'].insert(0, conclusions_cell)
    nb_json['cells'].insert(0, anomalies_cell)

    # Convert nb_json to nbformat object
    nb = nb_reads(json.dumps(nb_json), as_version=4)

    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {output_filename}")


def correct_text(text: str, speller=None) -> str:
    """
    Исправляет орфографические ошибки в тексте.

    Args:
        text (str): Текст, который нужно проверить.
        speller:  Объект, который будет использоваться для проверки орфографии.
            если None, то используется YandexSpeller()
    Returns:
        str: Исправленный текст.
    """
    if not speller:
        speller = YandexSpeller()
    # Используем регулярное выражение, чтобы найти все слова в тексте
    words = re.findall(r'\b\w+\b', text)

    # Создаем словарь, где ключами являются слова, а значениями - списки их позиций в тексте
    word_positions = {}
    for word in words:
        if word in word_positions:
            if word_positions[word][-1] + 1 < len(text):
                word_positions[word].append(
                    text.find(word, word_positions[word][-1] + 1))
            else:
                word_positions[word].append(text.find(word))
        else:
            word_positions[word] = [text.find(word)]
    # Удаляем все символы, кроме слов
    cleaned_text = re.sub(r'\W', '', text)
    # Исправляем ошибки в словах
    corrected_words = []
    max_attempts = 5  # Максимальное количество попыток
    attempts = 0
    text_has_errors = False
    while attempts < max_attempts:
        for word in words:
            # Исправляем ошибки в слове
            errors = speller.spell_text(word)
            if errors:
                text_has_errors = True
                text_for_highlight = list(text)
                highlight_word = f"<span style='color:yellow'>{word}</span>"
                # Если есть ошибки, берем первое предложенное исправление
                text_for_highlight = ''
                last_position = 0
                for start in word_positions[word]:
                    text_for_highlight += text[last_position:start] + \
                        highlight_word
                    last_position = start + len(word)
                text_for_highlight += text[last_position:]
                text_for_highlight = ''.join(text_for_highlight)
                display_html(text_for_highlight, raw=True)
                display_html(
                    f'Возможные верные варианты для слова {highlight_word}:', raw=True)
                print(errors[0]['s'])
                answer = input(
                    'Выберите индекс верного варианта (пустая строка - это индекс 0) или предложите свой вариант:\n')
                if answer.isdigit():
                    corrected_word = errors[0]['s'][int(answer)]
                elif answer == '':
                    corrected_word = errors[0]['s'][0]
                else:
                    corrected_word = answer

            else:
                corrected_word = word
            corrected_words.append(corrected_word)
        # print(words)
        # print(corrected_words)
        # Восстанавливаем слова на свои места
        if not text_has_errors:
            return text
        last_position = 0
        corrected_text = ''
        for i, word in enumerate(words):
            corrected_word = corrected_words[i]
            position = word_positions[word][0]
            word_positions[word].pop(0)
            corrected_text += text[last_position:position] + corrected_word
            last_position = position + len(word)
        corrected_text += text[last_position:]
        print('Исправленный вариант:')
        print(corrected_text)
        answer = input('Если верно, введите любой символ, -1 для повтора\n')
        if answer != '-1':
            break
        attempts += 1
    else:
        print("Максимальное количество попыток превышено. Исправленный текст не сохранен.")
        return text
    return corrected_text  # Объединяем символы обратно в строку


def correct_notebook_text(notebook_path: str, save_mode: str = 'final', work_mode: str = 'logging') -> None:
    """
    Corrects orthographic errors in a Jupyter Notebook.

    Args:
        notebook_path (str): Path to the Jupyter Notebook file.
        save_mode (str, optional): Mode of saving. Can be 'draft' or 'final'. Defaults to 'final'.
        work_mode (str, optional): Mode of working. Can be 'interactive', 'logging' or 'auto'. Defaults to 'logging'
    Returns:
        None
    """
    if save_mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid save_mode. save_mode must be either 'draft' or 'final'.")
    if work_mode not in ['interactive', 'logging', 'auto']:
        raise ValueError(
            "Invalid work_mode. work_mode must be one of 'interactive', 'logging', 'auto'")
    speller = YandexSpeller()
    # Используем регулярное выражение, чтобы найти все слова в тексте
    regex_for_find = re.compile(r'\b\w+\b')

    def correct_text(text: str) -> str:
        """
        Исправляет орфографические ошибки в тексте.

        Args:
            text (str): Текст, который нужно проверить.
        Returns:
            str: Исправленный текст.
        """
        # Проверим есть ли ошибки, если нет, то  вернем исходный текст
        # print('in correct_text')
        errors = speller.spell_text(text)
        if not errors:
            return text
        # Используем регулярное выражение, чтобы найти все слова в тексте
        words = regex_for_find.findall(text)

        # Создаем словарь, где ключами являются слова, а значениями - списки их позиций в тексте
        word_positions = {}
        for word in words:
            pattern = r'\b' + word + r'\b'
            indices = [m.start() for m in re.finditer(pattern, text)]
            word_positions[word] = indices
        # Исправляем ошибки в словах
        corrected_words = []
        max_attempts = 5  # Максимальное количество попыток
        attempts = 0
        while attempts < max_attempts:
            for word in words:
                # Исправляем ошибки в слове
                errors = speller.spell_text(word)
                if errors:
                    if work_mode in {'interactive', 'logging'}:
                        text_for_highlight = list(text)
                        highlight_word = f"<span style='color:yellow'>{word}</span>"
                        # Если есть ошибки, берем первое предложенное исправление
                        text_for_highlight = ''
                        last_position = 0
                        for start in word_positions[word]:
                            text_for_highlight += text[last_position:start] + \
                                highlight_word
                            last_position = start + len(word)
                        text_for_highlight += text[last_position:]
                        text_for_highlight = ''.join(text_for_highlight)
                        display_html(text_for_highlight, raw=True)
                        display_html(
                            f'Возможные верные варианты для слова {highlight_word}:', raw=True)
                        print(errors[0]['s'])
                        if work_mode == 'interactive':
                            answer = input(
                                'Выберите индекс верного варианта (пустая строка - это индекс 0) или предложите свой вариант:\n')
                            if answer.isdigit():
                                corrected_word = errors[0]['s'][int(answer)]
                            elif answer == '':
                                corrected_word = errors[0]['s'][0]
                            else:
                                corrected_word = answer
                        else:
                            corrected_word = errors[0]['s'][0]
                    else:
                        corrected_word = errors[0]['s'][0]
                else:
                    corrected_word = word
                corrected_words.append(corrected_word)
            # print(words)
            # print(corrected_words)
            # Восстанавливаем слова на свои места
            last_position = 0
            corrected_text = ''
            for i, word in enumerate(words):
                corrected_word = corrected_words[i]
                position = word_positions[word][0]
                word_positions[word].pop(0)
                corrected_text += text[last_position:position] + corrected_word
                last_position = position + len(word)
            corrected_text += text[last_position:]
            # Если нужно делать с подтверждением
            # print('Исправленный вариант:')
            # print(corrected_text)
            # answer = input('Если верно, введите любой символ, -1 для повтора\n')
            # if answer != '-1':
            #     break
            # attempts += 1
            break
        else:
            print(
                "Максимальное количество попыток превышено. Исправленный текст не сохранен.")
            return text
        return corrected_text  # Объединяем символы обратно в строку

    def read_notebook(notebook_path: str) -> dict:
        """
        Reads a Jupyter Notebook file and returns its contents as a JSON object.

        Args:
            notebook_path (str): Path to the Jupyter Notebook file.

        Returns:
            dict: Notebook contents as a JSON object.
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as in_f:
                return json.load(in_f)
        except FileNotFoundError:
            print(f"Error: File not found - {notebook_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format - {notebook_path}")
            return None

    def correct_source(source: list) -> list:
        """
        Corrects orthographic errors in a markdown cell.

        Args:
            source (list): Markdown cell contents.

        Returns:
            source: Corrected markdown cell contents.
        """
        corrected_lines = []
        for line in source:
            corrected_line = correct_text(line)
            corrected_lines.append(corrected_line)
        return corrected_lines

    def save_notebook(nb: dict, output_filename: str) -> None:
        """
        Saves a Jupyter Notebook to a file.

        Args:
            nb (dict): Notebook contents as a JSON object.
            output_filename (str): Output file name.

        Returns:
            None
        """
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            nb_write(nb, out_f, version=4)

    def is_markdown(it):
        return it["cell_type"] == "markdown"

    nb_json = read_notebook(notebook_path)
    if nb_json is None:
        return

    for cell in tqdm(filter(is_markdown, nb_json["cells"]), desc="Correcting cells", total=len(nb_json["cells"])):
        text = ' '.join(cell["source"])
        # print(text)
        if speller.spell_text(text):
            # print('before correct_source')
            cell["source"] = correct_source(cell["source"])

    if save_mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
        answer = input(
            'Был выбран режим работы "draft", результат сохраниться в файл "{output_filename}"\nЕсли хотите сохранить в исходный файл, то введите "final":\n')
        if answer == 'final':
            output_filename = notebook_path        
    else:
        output_filename = notebook_path
    print('End')
    nb = nb_reads(json.dumps(nb_json), as_version=4)
    save_notebook(nb, output_filename)

    print(f"Corrected notebook saved to {output_filename}")


def add_hypotheses_links_and_toc(notebook_path: str, mode: str = 'draft', link_type: str = 'html'):
    """
    Добавляет список гипотез в начало ноутбука Jupyter и ссылки на гипотезы. 

    Args:
        notebook_path (str): путь к ноутбуку Jupyter в формате JSON.
        mode (str, optional): режим работы функции. Defaults to 'draft'.
        link_type (str, optional): тип ссылок, которые будут добавлены в ноутбук. Defaults to 'html'.

    Raises:
        ValueError: если mode или link_type имеют недопустимые значения.
    """
    def is_markdown(it):
        return "markdown" == it["cell_type"]

    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    toc_hypotheses = []
    for cell in filter(is_markdown, nb_json["cells"]):
        source = cell["source"]
        new_source = []
        for line in source:
            if line.strip().startswith("_hypothesis_"):
                hypothesis = line.strip().replace("_hypothesis_ ", '')
                hypothesis_title = hypothesis.strip().strip('**')
                hypothesis_for_ref = regex_for_sub.sub(
                    '-', hypothesis_title)
                if link_type == "html":
                    toc_hypotheses.append(
                        f"- <a href='#{hypothesis_for_ref}'>{hypothesis_title}</a>  \n**Результат:**  \n")
                    hypothesis_for_ref = f"{hypothesis}<a name='{hypothesis_for_ref}'></a>"
                elif link_type == "markdown":
                    toc_hypotheses.append(
                        f"[- {hypothesis_title}](#{hypothesis_for_ref})  \n**Результат:**  \n")
                    hypothesis_for_ref = f"{hypothesis}<a class='anchor' id='{hypothesis_for_ref}'></a>"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
                hypothesis_for_ref += '  \n<a href="#ref-to-toc-hypotheses">Вернуться к выводам</a>'
                new_source.append(hypothesis_for_ref)
            else:
                new_source.append(line)
        cell["source"] = new_source
    toc_hypotheses = [
        '**Результаты проверки гипотез:**<a name="ref-to-toc-hypotheses"></a>\n\n',] + toc_hypotheses
    hypotheses_cell = nb_v4.new_markdown_cell([''.join(toc_hypotheses)])
    nb_json['cells'].insert(0, hypotheses_cell)
    # Convert nb_json to nbformat object
    nb = nb_reads(json.dumps(nb_json), as_version=4)
    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {output_filename}")

def check_observes_new_line(notebook_path: str, mode: str = 'draft'):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    for cell in nb_json["cells"]:
        source = cell["source"]
        new_source = []
        is_obs_found = False
        for line in source:
            # Если мы нашли "Наблюдения" и следующая строка не пустая, добавляем пустую строку
            if is_obs_found and line.strip() == '':
                is_obs_found = False  # Сбрасываем флаг
            elif is_obs_found:
                # Если это последняя строка и флаг установлен, добавляем пустую строку
                new_source.append('\n')  # Добавляем пустую строку         
            if '**Наблюдения:**' in line:
                is_obs_found = True
            new_source.append(line)
        cell["source"] = new_source    
    nb = nb_reads(json.dumps(nb_json), as_version=4)
    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {output_filename}")        
    
def collect_observations(notebook_path_for_find: str, notebook_path_for_save: str): #, mode: str = 'draft'):
    try:
        with open(notebook_path_for_find, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path_for_find}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path_for_find}")
        return
    is_working = False
    is_end = False
    observations = []
    for cell in nb_json["cells"]:
        source = cell["source"]
        is_obs_found = False
        for line in source:
            if '_pagristart_' in line:
                is_working = True
            if '_pagriend_' in line:
                is_working = False
                is_end = True
                break
            if is_working:
                # print(line)
                if is_obs_found and line.strip() != '':# Сбрасываем флаг
                    if not line.endswith('\n'):
                        line += '\n'
                    observations.append(line)
                if '**Наблюдения:**' in line:
                    is_obs_found = True
        if is_end:
            break
             
    observations_cell = nb_v4.new_markdown_cell([''.join(observations)])
    try:
        with open(notebook_path_for_save, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        return f"Error: File not found - {notebook_path_for_save}"
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format - {notebook_path_for_save}"    
    cell_number = next((i for i, cell in enumerate(nb_json["cells"])
                        if any("_gen_" in line for line in cell["source"])), None)
    nb_json['cells'][cell_number:cell_number] = [observations_cell]    
    
    nb = nb_reads(json.dumps(nb_json), as_version=4)
    # Save the nbformat object to the file
    # if mode == 'draft':
    #     output_filename_splited = notebook_path.split('.')
    #     output_filename_splited[-2] += '_temp'
    #     output_filename = '.'.join(output_filename_splited)
    # else:
    #     output_filename = notebook_path
    with open(notebook_path_for_save, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {notebook_path_for_save}")            

def add_conclusions(notebook_path: str, conclusions: list, mode: str = 'draft', link_type: str = 'html'):
    """
    This function adds conclusions and anomalies sections to a Jupyter notebook.

    Args:
        notebook_path (str): The path to the Jupyter notebook file.
        mode (str): The mode of the output file, either 'draft' or 'final'. Defaults to 'draft'.
        link_type (str): The type of link to use, either 'html' or 'markdown'. Defaults to 'html'.
        order (dict): dict of lists with ordered  conclusions and anomalie (key: conclusions and anomalies)

    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            notebook = nb_read(in_f, as_version=4)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    ordered_conclusions = [line.strip(" -.,") for line in conclusions]
    conclusions = set(line.strip(" -.,") for line in conclusions)
    conclusions_added = set()
    insertions_and_new_cells = [] 
    cell_has_ref_to_toc = False
    conclusions_refs = []
    # Проходим по ячейкам в обратном порядке
    for i in range(len(notebook.cells)):
        cell = notebook.cells[i]
        source = cell["source"]
        cell_has_ref_to_toc = False
        for conclusion in conclusions:
            if conclusion in source and conclusion not in conclusions_added:
                conclusions_added.add(conclusion)
                conclusion_for_ref = regex_for_sub.sub(
                    '-', conclusion)
                if link_type == "html":
                    toc_conclusion = f"- <a href='#{conclusion_for_ref}'>{conclusion+'.'}</a>  \n"
                elif link_type == "markdown":
                    toc_conclusion = f"[- {conclusion+'.'}](#{conclusion_for_ref})  \n"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
                if link_type == "html":
                    conclusion_for_ref = f"<a name='{conclusion_for_ref}'></a>"
                elif link_type == "markdown":
                    conclusion_for_ref = f"<a class='anchor' id='{conclusion_for_ref}'></a>"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'name' или 'id'.")

                conclusions_refs.append((toc_conclusion, conclusion))
                if not cell_has_ref_to_toc:
                    conclusion_for_ref += '\n<a href="#ref-to-conclusions">Вернуться к выводам</a>'
                new_cell = nb_v4.new_markdown_cell(conclusion_for_ref)            
                cell_has_ref_to_toc = True
                insertions_and_new_cells.append((i - 1, new_cell))

    # Вставляем ячейки в обратном порядке, чтобы не нарушить индексы
    for index, cell in reversed(insertions_and_new_cells):
        notebook.cells.insert(index, cell)    
    # Отсортируем  выводы и аномалии как в переданном словаре
    index_map = {x: i for i, x in enumerate(ordered_conclusions)}
    conclusions_refs = sorted(
        conclusions_refs, key=lambda x: index_map[x[1]])
    conclusions_refs = [conclusions_ref[0] for conclusions_ref in conclusions_refs]
    # Добавим содержание выводов
    conclusions_refs = [
        '**Главные выводы:**<a name="ref-to-conclusions"></a>  \n'] + conclusions_refs
    # anomalies = [
    #     '**Аномалии и особенности в данных:**<a name="ref-to-anomalies"></a>  \n'] + anomalies
    conclusions_cell = nb_v4.new_markdown_cell([''.join(conclusions_refs)])
    # anomalies_cell = nb_v4.new_markdown_cell([''.join(anomalies)])    
    notebook.cells.insert(0, conclusions_cell) 
    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nb_write(notebook, out_f, version=4)
    print(f"Corrected notebook saved to {output_filename}")
    residuals = conclusions.difference(conclusions_added)
    if residuals:
        print("Не удалось добавить все выводы. Не удалось добавить:")
        for res in residuals:
            print(res)
              
def split_text_to_cells(notebook_path_for_find: str, notebook_path_for_save: str): #, mode: str = 'draft'):
    try:
        with open(notebook_path_for_find, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path_for_find}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path_for_find}")
        return
    is_working = False
    is_end = False
    split_lines = []
    for cell in nb_json["cells"]:
        source = cell["source"]
        for line in source:
            if '_pagristart_' in line:
                is_working = True
            if '_pagriend_' in line:
                is_working = False
                is_end = True
                break
            if is_working:
                # print(line)
                split_lines.append(line)
        if is_end:
            break
    new_cells = []
    for line in split_lines:         
        new_cells.append(nb_v4.new_markdown_cell([line]))
    try:
        with open(notebook_path_for_save, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        return f"Error: File not found - {notebook_path_for_save}"
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format - {notebook_path_for_save}"    
    cell_number = next((i for i, cell in enumerate(nb_json["cells"])
                        if any("_gen_" in line for line in cell["source"])), None)
    nb_json['cells'][cell_number:cell_number] = new_cells    
    
    nb = nb_reads(json.dumps(nb_json), as_version=4)
    # Save the nbformat object to the file
    # if mode == 'draft':
    #     output_filename_splited = notebook_path.split('.')
    #     output_filename_splited[-2] += '_temp'
    #     output_filename = '.'.join(output_filename_splited)
    # else:
    #     output_filename = notebook_path
    with open(notebook_path_for_save, 'w', encoding='utf-8') as out_f:
        nb_write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {notebook_path_for_save}")                   
    
def add_dataframe_columns_exploration(notebook_path: str, df_name: str, df, heading_level: int = 4):
    """
    Добавляет в Jupyter notebook ячейки для анализа каждой колонки датафрейма.
    
    Args:
        notebook_path (str): Путь к файлу .ipynb
        df_name (str): Имя датафрейма как строка (для использования в коде)
        df: Сам датафрейм (используется только для получения списка колонок)
        heading_level (int): Уровень заголовка (количество '#')
    """
    try:
        # Читаем существующий notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nb_read(f, as_version=4)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    
    # Создаем новые ячейки для каждой колонки
    new_cells = []
    
    header_cell = nb_v4.new_markdown_cell(
        f"{'#' * heading_level}  Таблица {df_name}"
    )
    code_cell_1 = nb_v4.new_code_cell(
        f"{df_name}.head(1)"
    )
    cell_1 = nb_v4.new_markdown_cell(
        "Посмотрим на информацию о датафрейме."
    )
    
    code_cell_2 = nb_v4.new_code_cell(
        f"{df_name}.explore.info()"
    )
    
    cell_2 = nb_v4.new_markdown_cell(
        "Изучим по отдельности каждый столбец на пропуски, выбросы и прочие аномалии."
    )
    new_cells.extend([header_cell, cell_1, code_cell, cell_2])
    for column in df.columns:
        # 1. Ячейка с заголовком
        header_cell = nb_v4.new_markdown_cell(
            f"{'#' * (heading_level+1)} {column}"
        )
        
        # 2. Ячейка с кодом для анализа
        code_cell = nb_v4.new_code_cell(
            f"{df_name}['{column}'].explore.info()"
        )
        
        # 3. Ячейка с наблюдениями
        observations_cell = nb_v4.new_markdown_cell(
            "**Наблюдения:**  \n\n"
            "- Комментарии\n"
        )
        
        new_cells.extend([header_cell, code_cell, observations_cell])
    
    # Вставляем новые ячейки в начало notebook
    nb.cells = new_cells + nb.cells
    
    # Сохраняем изменения
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nb_write(nb, f)
    
    print(f"Added exploration cells for {len(df.columns)} columns to {notebook_path}")    