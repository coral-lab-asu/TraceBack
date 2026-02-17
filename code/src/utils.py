import re

text = """<Final Answer>
1. [(1,0),(1,1)],[(13,0),(13,1)]
2. [(1,0),(1,1)]

Output Explanation:
1. For the first sub-question, the Free Democratic Party, located in row 1, has the most seats: 49. The table shows that the total number of seats is 187, which is found in row 13, column 1.
2. For the second sub-question, the number of seats won by the Free Democratic Party is located in row 1, column 1, which is 49. The party's name is in row 1, column 0.
"""
text = """<Final Answer>
1. [(1,0),(1,1),(13,0),(13,1)]
2. [(1,0),(1,1)]

Output Explanation:
1. For the first sub-question, the Free Democratic Party, located in row 1, has the most seats: 49. The table shows that the total number of seats is 187, which is found in row 13, column 1.
2. For the second sub-question, the number of seats won by the Free Democratic Party is located in row 1, column 1, which is 49. The party's name is in row 1, column 0.
"""
# Regular expression to match lines like:
# 1. [(10,0)]
# 2. []
def parse_attribution(text):
    pattern = re.compile(r'^(\d+)\.\s*\[(.*?)\]$')

    final_answer = []

    # Split into lines and search for matches
    for line in text.splitlines():
        line = line.strip()
        match = pattern.search(line)

        if match:

            content = match.group(2).strip()  # The part inside [ ... ]
            content = '['+content+']'
            if content:
                lists = eval(content)
                for l in lists:
                    if isinstance(l, list):
                        final_answer.extend(l)
                    else:
                        final_answer.append(l)
                
            else:
                # If there's no content (i.e. an empty []), just store an empty tuple
                final_answer.append(())
    print(final_answer)
parse_attribution(text)
# Output: [(10, 0), (1, 1), (2, 1), (), ()]
