import re

def word_count(m):
    words = m.strip().split(" ")
    return len(words)

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

def chunks_by_content(filepath:str) -> list[str]:
    files = []
    titles = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    sections = content.split("—--------------------")

    for s in sections:
        small_parts = [line for line in s.split("\n") if line.strip()]
        final_parts = []
        for k in small_parts:
            if word_count(k) < 150:
                final_parts.append(k)
            else:
                sentences = k.split(".")
                final_parts.extend(sentences)
        title = small_parts[0]
        titles.append(title)

    for idx in range(0,len(sections)):
        text = sections[idx]
        if idx == 0:
            pattern = r"^(.*?)(Overview:.*?)(Our Approach:.*)$"
            matches = re.search(pattern, text, re.DOTALL)
            chunks = [matches.group(i).strip() for i in range(1,matches.re.groups+1)]
            for chunk in chunks:
                final_text = f"{titles[idx]}\n\n{chunk}"
                files.append(final_text)
        if idx == 1:
            pattern = r"^(.*?)(Steps:.*?)(Notes:.*?)(Yield Source:.*?)(Some key notes worth noting:.*?)$"
            matches = re.search(pattern, text, re.DOTALL)
            chunk_1 = [matches.group(i).strip() for i in range(2,matches.re.groups+1)]
            for c in chunk_1:
                final_text = f"{titles[idx]}\n\n{c}"
                files.append(final_text)
        if idx == 2:
            pattern = r"^(.*?)(Steps:.*?)(Notes:.*?)(Yield Source:.*?)(Key things to watch:.*?)$"
            matches = re.search(pattern, text, re.DOTALL)
            chunk_2 = [matches.group(i).strip() for i in range(2,matches.re.groups+1)]
            for c in chunk_2:
                final_text = f"{titles[idx]}\n\n{c}"
                files.append(final_text)
    return files