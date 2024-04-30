##################### new updated code ##############################
# Add custom CSS to hide the GitHub icon
import streamlit as st
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from dbconnection import pipeline_qa_chain

st.title("DB Analyst :robot_face:")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome to the AI :wave:"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # OPENAI BOT PROCESSING RESPONSE
    with st.status(
        "AI is thinking..:grey_exclamation: :brain:", state="running"
    ):
        print(user_input)
        response= pipeline_qa_chain(user_input)
        print(response)

    # Add the assistant's response to the chat messages
    st.session_state.messages.append(
        {"role": "assistant", "content": response}  
    )

    # Write individual elements
    st.chat_message("assistant").write(response)
    # if df is not None:
    #     st.dataframe(df)
    # if pie_chart is not None:
    #     st.plotly_chart(pie_chart, use_container_width=True)
    # if bar_chart is not None:
    #     st.plotly_chart(bar_chart, use_container_width=True)





