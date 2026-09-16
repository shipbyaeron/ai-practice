import re

def word_count(m):
    words = m.strip().split(" ")
    return len(words)

def portion_split(paragraph: str) -> list[str]:
    final_split = []
    WORD_LIMIT = 150
    sentences = paragraph.split(".")
    len_sen = len(sentences)
    i = 0
    while i < len_sen:
        current_group = [sentences[i]]
        current_length = word_count(sentences[i])
        j = i + 1
        while j < len_sen:
            next_length = word_count(sentences[j])
            if current_length + next_length > WORD_LIMIT:
                break
            current_group.append(sentences[j])
            current_length += next_length
            j += 1
        final_text = " ".join(current_group)
        final_split.append(final_text)
        i = j
    return final_split

def build_chunks(filepath: str) -> list[str]:
    files = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    sections = content.split("—--------------------")

    for s in sections:
        small_chunks = [line for line in s.split("\n") if line.strip()]
        final_chunks = []
        for k in small_chunks:
            if word_count(k) < 150:
                final_chunks.append(k)
            else:
                sentences = k.split(".")
                final_chunks.extend(sentences)
        title = small_chunks[0]
        len_small = len(final_chunks)
        i = 1
        while i < len_small:
            current_group = [final_chunks[i]]
            current_length = word_count(final_chunks[i])
            j = i + 1
            while j < len_small:
                next_length = word_count(final_chunks[j])
                if current_length + next_length > 150:
                    break
                current_group.append(final_chunks[j])
                current_length += next_length
                j += 1
            merged_text = " ".join(current_group)
            text_to_embedded = f"{title}\n\n{merged_text}"
            files.append(text_to_embedded)
            i = j
    return files

def chunks_by_section(filepath:str) -> list[str]:
    chunks = []
    WORD_LIMIT = 150
    OVER_LAP = 1
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    para = [line for line in content.split("\n\n") if line.strip()]
    article_title = para[0]

    sections = content.split("—--------------------")

    for section in sections:
        if section.strip():
            small_parts = [line for line in section.split("\n") if line.strip()]
            final_parts = []
            for k in small_parts:
                if word_count(k) <= WORD_LIMIT:
                    final_parts.append(k)
                else:
                    split_k = portion_split(k)
                    final_parts.extend(split_k)
            title = final_parts[0]
            section_length = len(final_parts)
            i = 1
            while i < section_length:
                current_group = [final_parts[i]]
                current_length = word_count(final_parts[i])
                j = i + 1
                move = 0
                while j < section_length:
                    next_length = word_count(final_parts[j])
                    if current_length + next_length > WORD_LIMIT:
                        break
                    current_group.append(final_parts[j])
                    current_length += next_length
                    move += 1
                    j += 1
                merged_text = " ".join(current_group)
                text_to_embedded = f"{title}\n\n{merged_text}"
                chunks.append(text_to_embedded)
                if move < OVER_LAP:
                    i = j
                else:
                    i = j - OVER_LAP
    return (article_title, chunks)