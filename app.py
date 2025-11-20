import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd
import re

st.set_page_config(page_title="Langchain Chat with SQL DB", page_icon="🦜")
st.title("🦜 Langchain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
POSTGRESQL = "USE_POSTGRESQL"

radio_opt = ["Use SQLLite 3 Database Student.db", "Connect to your POSTGRESQL Database"]

selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_url = POSTGRESQL
    postgresql_host = st.sidebar.text_input("Provide POSTGRESQL Host")
    postgresql_user = st.sidebar.text_input("POSTGRESQL User")
    postgresql_password = st.sidebar.text_input("POSTGRESQL password", type="password")
    postgresql_db = st.sidebar.text_input("POSTGRESQL database")
else:
    db_url=LOCALDB

api_key = st.sidebar.text_input(label="GROQ_API_KEY", type="password")

if not api_key:
    st.info("Please add the groq api key")
    st.stop()

if db_url == POSTGRESQL:
    if not (postgresql_host and postgresql_user and postgresql_password and postgresql_db):
        st.info("Please provide all PostgreSQL connection details.")
        st.stop()

## LLM model - only create if API key exists
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_url, postgresql_host=None, postgresql_user=None, postgresql_password=None, postgresql_db=None):
    if db_url == LOCALDB:
        dbfilepath = (Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_url==POSTGRESQL:
        if not (postgresql_host and postgresql_user and postgresql_password and postgresql_db):
            st.error("Please provide all POSTGRESQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"postgresql://{postgresql_user}:{postgresql_password}@{postgresql_host}/{postgresql_db}"))

if db_url == POSTGRESQL:
    db = configure_db(db_url, postgresql_host, postgresql_user, postgresql_password, postgresql_db)
else:
    db = configure_db(db_url)

## toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        
        # Function to parse concatenated database results
        def parse_database_response(response_text):
            try:
                # Remove extra whitespace and split into potential records
                clean_text = response_text.strip()
                
                # Check for tuple list format: [('Krish', 'Data Science', 'A', 90), ...]
                if clean_text.startswith('[') and clean_text.endswith(']'):
                    try:
                        # Use eval to parse the tuple list (safe since it's our own data)
                        tuple_list = eval(clean_text)
                        if isinstance(tuple_list, list) and len(tuple_list) > 0:
                            if isinstance(tuple_list[0], tuple) and len(tuple_list[0]) >= 4:
                                rows = []
                                for item in tuple_list:
                                    if len(item) >= 4:
                                        rows.append([str(item[0]), str(item[1]), str(item[2]), str(item[3])])
                                
                                if rows:
                                    df = pd.DataFrame(rows, columns=['NAME', 'CLASS', 'SECTION', 'MARKS'])
                                    return df
                    except:
                        pass
                
                # Check for comma-separated values
                if ',' in clean_text and not clean_text.startswith('['):
                    # Split by commas and group into sets of 4
                    parts = [part.strip() for part in clean_text.split(',')]
                    if len(parts) >= 4 and len(parts) % 4 == 0:
                        rows = []
                        for i in range(0, len(parts), 4):
                            if i + 3 < len(parts):
                                name = parts[i]
                                class_name = parts[i+1]
                                section = parts[i+2]
                                marks = parts[i+3]
                                
                                # Validate that marks is numeric
                                if marks.isdigit():
                                    rows.append([name, class_name, section, marks])
                        
                        if rows:
                            df = pd.DataFrame(rows, columns=['NAME', 'CLASS', 'SECTION', 'MARKS'])
                            return df
                
                # Check if it looks like concatenated database records (space-separated)
                # Pattern: Name Subject Grade Number (repeating)
                pattern = r'([A-Za-z]+)\s+([A-Za-z\s]+)\s+([A-Z])\s+(\d+)'
                matches = re.findall(pattern, clean_text)
                
                if matches and len(matches) > 1:
                    # Create DataFrame from matches
                    df = pd.DataFrame(matches, columns=['NAME', 'CLASS', 'SECTION', 'MARKS'])
                    return df
                
                # Alternative: Try to split by common patterns for space-separated
                # If response looks like space-separated values in sequence
                words = clean_text.split()
                if len(words) >= 8 and len(words) % 4 == 0:  # Multiple of 4 (4 columns)
                    rows = []
                    for i in range(0, len(words), 4):
                        if i + 3 < len(words):
                            # Handle multi-word class names like "Data Science"
                            name = words[i]
                            
                            # Look for the section (single letter) and marks (number)
                            section = ''
                            marks = ''
                            class_parts = []
                            
                            for j in range(i+1, min(i+4, len(words))):
                                if words[j].isdigit() and not marks:
                                    marks = words[j]
                                elif len(words[j]) == 1 and words[j].isalpha() and not section:
                                    section = words[j]
                                else:
                                    class_parts.append(words[j])
                            
                            class_name = ' '.join(class_parts)
                            
                            if marks.isdigit() and section and class_name:
                                rows.append([name, class_name, section, marks])
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=['NAME', 'CLASS', 'SECTION', 'MARKS'])
                        return df
                
                # Try parsing line by line if newlines exist
                lines = clean_text.split('\n')
                if len(lines) > 1:
                    rows = []
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Check if line is comma-separated
                            if ',' in line:
                                parts = [part.strip() for part in line.split(',')]
                                if len(parts) >= 4:
                                    rows.append(parts[:4])  # Take first 4 parts
                            else:
                                # Try to parse each line (space-separated)
                                parts = line.split()
                                if len(parts) >= 4:
                                    name = parts[0]
                                    marks = parts[-1] if parts[-1].isdigit() else ''
                                    section = parts[-2] if len(parts[-2]) == 1 and parts[-2].isalpha() else ''
                                    class_name = ' '.join(parts[1:-2]) if section and marks else ' '.join(parts[1:-1])
                                    
                                    if marks.isdigit():
                                        rows.append([name, class_name, section, marks])
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=['NAME', 'CLASS', 'SECTION', 'MARKS'])
                        return df
                        
            except Exception as e:
                pass
            
            return None
        
        # Function to check if response contains data that can be shown as table
        def format_response_as_table(response_text):
            try:
                # First try to parse as database response
                db_df = parse_database_response(response_text)
                if db_df is not None:
                    return db_df
                
                # Original table parsing logic
                lines = response_text.strip().split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                
                has_multiple_rows = len(lines) > 1
                has_separators = any('|' in line or ',' in line or '\t' in line for line in lines)
                
                if has_multiple_rows and has_separators:
                    data_rows = []
                    headers = None
                    
                    for i, line in enumerate(lines):
                        if re.match(r'^[\s\|\-\+]+$', line):
                            continue
                            
                        if '|' in line:
                            row = [item.strip() for item in line.split('|') if item.strip()]
                        elif '\t' in line:
                            row = [item.strip() for item in line.split('\t')]
                        elif ',' in line and len(line.split(',')) > 1:
                            row = [item.strip() for item in line.split(',')]
                        else:
                            continue
                            
                        if row:
                            if headers is None and i == 0:
                                headers = row
                            else:
                                data_rows.append(row)
                    
                    if data_rows:
                        if headers and len(headers) == len(data_rows[0]):
                            df = pd.DataFrame(data_rows, columns=headers)
                        else:
                            df = pd.DataFrame(data_rows)
                        return df
                
                # Handle single values
                if len(lines) == 1:
                    line = lines[0]
                    if (line.replace('.', '').replace('-', '').replace('/', '').replace(':', '').isdigit() or
                        len(line.split()) <= 3 or
                        re.match(r'^\d+$', line) or
                        re.match(r'^\d+\.\d+$', line) or
                        re.match(r'^\d{4}-\d{2}-\d{2}', line) or
                        line.lower() in ['yes', 'no', 'true', 'false']):
                        
                        df = pd.DataFrame([line], columns=['Result'])
                        return df
                        
            except Exception as e:
                pass
            
            return None
        
        # Try to format as table
        table_df = format_response_as_table(response)
        
        if table_df is not None:
            st.dataframe(table_df, use_container_width=True)
        else:
            st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
