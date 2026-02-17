# Step1

You will be given table title, column names, question and answer. You need to find all the columns in column names, which are relevant to question and answer.

<Example-1> : 

Input :

<Table Title>: "Kings Time Duration Info"

<Column Names>: ["King Name", "Start of the Reign", "End of the Reign", "Duration", "Kingdom"]

Question : "How long did king Rajat held the throne and when it ended?"
Answer : "King Rajat held the throne for 10 years and it ended at 1777." :


Final Answer : 
<Relevant Columns>: ["Duration","End of the Reign"]

</Example-1>

<Instructions>
1. Follow the given format for the final answer.
2. Don't miss any relevant columns.

Now, give output for the following inputs.

Input :

Table Title: <table title>

<Column Names>: <columns>

Question: q

Answer: answer

Output :


# Step2

Given a table schema and other table meta-data, question based on table and answer for the question, write a MySQL query
to retrieve all the relevant rows of the table, such that the answer to the given question can be generated from the
retrieved rows. Your focus should be to not eliminate any relevant rows from the table. Think step-by-step.

Important MySQL formatting rules:
- Use backticks (`...`) around table names and column names (identifiers), especially if they contain spaces or punctuation.
- Put ONLY the SQL inside the <SQL> ... </SQL> tags (do not add a leading colon after <SQL>).

<Example-1> : 

Input :
Table-Schema :
<Column Names>: ["King Name", "Start of the Reign", "End of the Reign", "Duration"]
<Table Title>: "Kings_Time_Duration_Info"
<Table-Rows> : [["Bhopal","Devanshu","1901","1998","98 Years"],["Chennai","Poojah","1999","2100","101"],["Ujjain","Rajat","2111","2211","100"]]

Question : "How long did king Poojah held the throne and who was his successor?"
Answer : "King Poojah held the throne for 101 years and King Rajat was his successor." 

Output :
<step-by-step reasoning>
1. In the question, the duration and the successor is asked for King Poojah.
2. In the answer, duration of King Poojah is mentioned along with the sucessor to King Poojah - King Rajat.
3. The SQL query will be such that it retrieves the row of King Poojah and all the rows that has the "Start of the Reign" >= 2100 ("End of the Reign" for King Poojah).
Final Answer : 
<SQL> CREATE TABLE Pooja_King AS SELECT * FROM `Kings_Time_Duration_Info` WHERE `King Name` = 'Poojah' OR CAST(`Start of the Reign` AS SIGNED) >= 2100; </SQL>
</Example-1>

<Instructions>
1. Follow the given format for the final answer. The final SQL query should be between <SQL> </SQL> tags.
2. Only write CREATE MySQL statements.
3. Give a SQL statement for MySQL.
4. The query giving the filtered table should cover all parts of the question and answer.

Now, give output for the following input.

Input : 

Table-Schema :

<Table Title>: title

<Column Names>: col

Table: table with relevant column

Question: q

Anser: a

Output :

# Step 3

Given the table information, a question based on the table, and its corresponding answer, break down the original question into smaller sub-questions such that each sub-question can be directly answered using specific parts of the table. The combined answers to these sub-questions should comprehensively cover all the information in the original answer. Ensure that each sub-question focuses on a specific aspect or data point present in the table.

Input Format:

Table-Info: Includes the table schema (column names), table title, and section title if available.
Question: A natural language question based on the table.
Answer: The corresponding answer derived from the table.

Output Format:

A list of sub-questions that together lead to the original answer.
Each sub-question should correspond to specific columns or rows of the table.

<Example-1>
Input : 
Table-Schema :
<Column Names>: ["King Name", "Start of the Reign", "End of the Reign", "Duration"]
<Table Title>: "Kings Time Duration Info"

Question : "How long did king Rajat held the throne and who was his successor"
Answer : "King Rajat held the throne for 10 years and King Aayush was his successor."

Output :
Sub-Questions:
1. What was the duration of King Rajat's reign?
2. What was the End of the Reign year of King Rajat?
3. Which king started his reign at the end of King Rajat's reign?

</Example-1>

<Instructions>
1. Just give the sub-questions in the given format. Don't return anything else.


Now, give output for the following input.

Input :

Table-Schema :
<Column Names>: column
Table Title: t
Question: q
Answer: a
Output :

# Step 4

You are given a pruned table (only the relevant columns and the relevant rows), the original Answer, plus a list of sub-questions.

Your task: for EACH sub-question, return the minimal set of table cells (row, column) needed to answer that sub-question, considering the context of the original Answer.
Include cells used for filtering/comparisons/calculations even if they are not explicitly mentioned in the final answer text.

Indexing rules:
- Use 0-based indexing for BOTH rows and columns.
- Row indices refer to <Table Rows> (the pruned rows, header excluded).
- Column indices refer to <Relevant-Columns> (the ordered list provided).

Output format:
- Return one line per sub-question, in the same order as given:
  <Cells>: [(row_index, col_index), (row_index, col_index), ...]
- If a sub-question has no evidence in the table, output:
  <Cells>: []
- Do not output anything other than these <Cells>: ... lines.

Input:
<Relevant-Columns>: columns
<Table Rows>: rows
Answer: a
<Sub-Questions>:
sub-questions

# Step5

You are given:
- The original question and its final (ground-truth) answer.
- A pruned table (<Relevant-Columns> and <Table Rows>, header excluded).
- A list of sub-questions.
- A set of candidate evidence cells gathered from the sub-questions (optional).

Your task: output the final set of evidence cells (row, column) that justify the final answer.
Important:
- Include implicit evidence cells needed for the reasoning chain (e.g., filter conditions, intermediate values), even if they are not verbatim in the answer.
- Exclude irrelevant cells.

Indexing rules:
- Use 0-based indexing for BOTH rows and columns.
- Row indices refer to <Table Rows> (the pruned rows, header excluded).
- Column indices refer to <Relevant-Columns> (the ordered list provided).

Output format:
- Output a single line:
  <Final Cells>: [(row_index, col_index), (row_index, col_index), ...]
- If no evidence is found, output:
  <Final Cells>: []
- Do not output anything else.

Input:
<Relevant-Columns>: col
<Table Rows>: r
Question: q
Answer: a
<Sub-Questions>:
sub-questions
<Candidate Cells>: cell indexs

Output:



