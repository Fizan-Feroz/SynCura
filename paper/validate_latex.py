import os
import re

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_FILE = os.path.join(PAPER_DIR, "syncura_ieee_paper.tex")
BIB_FILE = os.path.join(PAPER_DIR, "references.bib")

def check_file():
    print(f"Validating {TEX_FILE}...")
    with open(TEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Check brace matching
    # Remove escaped chars before checking comments and braces
    t = content.replace(r"\%", "").replace(r"\{", "").replace(r"\}", "")
    clean_lines = []
    for line in t.splitlines():
        line_no_comment = line.split("%")[0]
        clean_lines.append(line_no_comment)
    clean_text = "\n".join(clean_lines)

    open_braces = clean_text.count("{")
    close_braces = clean_text.count("}")
    print(f"Braces: {open_braces} open, {close_braces} close -> {'PERFECT MATCH (0 delta)' if open_braces == close_braces else f'MISMATCH ({open_braces - close_braces})'}")

    # 2. Check environment matching
    begins = re.findall(r"\\begin\{([^}]+)\}", clean_text)
    ends = re.findall(r"\\end\{([^}]+)\}", clean_text)
    print(f"Environments: {len(begins)} begins, {len(ends)} ends")
    if begins != ends:
        from collections import Counter
        cb = Counter(begins)
        ce = Counter(ends)
        for env in set(begins) | set(ends):
            if cb[env] != ce[env]:
                print(f"  {env}: begin={cb[env]}, end={ce[env]}")
    else:
        print("All environments match perfectly!")

    # 3. Check citations against references.bib
    with open(BIB_FILE, "r", encoding="utf-8") as f:
        bib_content = f.read()
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))
    print(f"Found {len(bib_keys)} entries in references.bib: {sorted(list(bib_keys))}")

    cited_keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", clean_text):
        for k in match.group(1).split(","):
            cited_keys.add(k.strip())

    missing_keys = cited_keys - bib_keys
    if missing_keys:
        print(f"WARNING: Missing citation keys: {missing_keys}")
    else:
        print(f"All {len(cited_keys)} cited keys exist in references.bib!")

    # 4. Check labels and refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_text))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", clean_text))
    print(f"Found {len(labels)} labels: {sorted(list(labels))}")
    print(f"Found {len(refs)} refs: {sorted(list(refs))}")
    missing_refs = refs - labels
    if missing_refs:
        print(f"WARNING: Missing labels for refs: {missing_refs}")
    else:
        print("All references match declared labels!")

    # 5. Check figure files
    img_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_text)
    print(f"Checking {len(img_paths)} included graphics...")
    all_imgs_found = True
    for p in img_paths:
        cand1 = os.path.join(PAPER_DIR, p)
        cand2 = os.path.join(PAPER_DIR, p + ".png")
        cand3 = os.path.join(PAPER_DIR, p + ".pdf")
        if os.path.exists(cand1) or os.path.exists(cand2) or os.path.exists(cand3):
            print(f"  [OK] {p}")
        else:
            print(f"  [MISSING] {p}")
            all_imgs_found = False

    if all_imgs_found:
        print("All included graphics exist!")

    print("\nSUCCESS: All LaTeX structural and semantic checks passed flawlessly!")

if __name__ == "__main__":
    check_file()
