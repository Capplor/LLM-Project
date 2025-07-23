import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langsmith import Client
from langsmith import traceable
import os

from helper_prompts import *


# Langsmith set-up 
smith_client = Client()

st.set_page_config(page_title="Step 3: design your example scenario", page_icon="🤷🏻‍♂️")
# st.title("Step 1: Design your questions")




@traceable
def create_scenario(history,example_set,persona):
    chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key = st.secrets.openai_api_key)

    # start by setting up the langchain chain from our template (defined in lc_prompts.py)
    scenario_template = PromptTemplate(input_variables = ["history", "example_set", "persona"], template = generation_prompt)
    # st.write(scenario_template)

    # add a json parser to make sure the output is a json object
    json_parser = SimpleJsonOutputParser()

    # connect the prompt with the llm call, and then ensure output is json with our new parser
    chain = scenario_template | chat | json_parser

    scenario = chain.invoke({"history" : history, "persona" : persona, "example_set" : example_set})

    return(scenario)




def setupSidebar(): 
    ## set up side bar content
    st.sidebar.markdown("Your goal for this page are to finalise your example answers and create a scenario based on them. This will be guiding all the narrative creation that your bot does in the future, so is critical!")


def toggle_ready():
    st.session_state['ready'] = not st.session_state['ready'] 



# Define your CSS and inject it -- ensuring smaller font in the text areas. 
style = """
<style>
.stTextArea>div>div>textarea {
    font-size: 12px;
}
</style>
"""
st.markdown(style, unsafe_allow_html=True)

setupSidebar()


## set up basic structure of the page: 
# Prepare two tabs, one tab to explore in, the other to finalise and move on. 
tab1, tab2, tab3 = st.tabs(['👷‍♀️ Build your example', '🔎 Update your example', '🔮 See current configuration'])


## First working with tab1 -- helping the user identify their questions
with tab1: 
    st.write("""Great, you've decided on questions and tried how they work. This next step allows you to test your example answers and create an example scenario for your bot.
1. Check the 'extracted' dialogue from the previous step, and adapt if needed.  
2. Test how a resulting narrative could look like by pressing Ready
3. Finalise and update the example scenario in the next tab.""")

    exp_qs = st.expander("**Insert your example answers here and press Ready when done**", expanded = False)
    exp_llm = st.container()  # insert container to show the LLM in

# write the recent value of convo 
with exp_qs:
    st.text_area("fill in your conversation here:", value = st.session_state.package['convo_extract'], height= 400, key = "convo_history", label_visibility= "hidden")
    st.button("Ready!", key = "ready_llm")


## check that we have the button visualised 
if 'ready_llm' in st.session_state: 
    ## now wait for it to be pressed:
    if st.session_state['ready_llm']:

        # once pressed, let's save the current convo  first. Note that we are using the persona from the package directly: 
        st.session_state.package['convo_extract'] = st.session_state['convo_history']
        st.session_state['persona'] = st.session_state.package['persona']

        # load up the data from package: 
        lines = st.session_state.package['questions_str'].splitlines()
        nq = st.session_state.package['questions_num']
        with exp_llm:
            st.write("**This is what LLM extracts:**")
            
            ## create the JSON keys / questions: 
            question_pairs = {}
            for n in range (1, nq + 1):
                question_pairs[f'Q{n}'] = f"Q{n}: {lines[n-1].strip()}"

            # Collecting all keys and values in separate lists
            json_keys = list(question_pairs.keys())
            questions = list(question_pairs.values())

            with st.spinner('Creating your scenario...'):
                scenario = create_scenario(st.session_state['convo_history'], st.session_state.package['current_example_set'], st.session_state['persona'])

            if 'new_scenario' in scenario:  # assuming that the LLM call went well!
                # save the current version of the example in the config-package
                example_set = f"{st.session_state['convo_history']} \n \n The scenario based on these responses: {scenario['new_scenario']}"
                st.session_state.package['new_example_set'] = example_set
                
                # show to the user 
                st.write(scenario['new_scenario'])
            else: 
                st.write("Error in scenario generation: ", scenario)
            # st.write(choices)

## Moving onto the second tab -- now adapting & finalising the scenario examples. 
with tab2: 
    st.write("This tab will help you finalise your example answers set, based on your questions. This will guide your bot as the _'One shot'_ example -- so it is crucial that you have it right, and take a neutral tone.")

    st.write("This is what you have so far ... feel free to edit the answers & resulting scenario, but **do not alter the questions themselves** -- you would need to go back to the `Build your Qs' tab to do that.")

    st.text_area("Finalise your example answers set here:", value = st.session_state.package['new_example_set'], height= 400, key = "fix_answer_set", label_visibility= "hidden") 

    st.button("Save the example above into the bot", key = "fix_answer_set_button")

    # if update clicked  
    if st.session_state['fix_answer_set_button']:
        st.session_state.package['current_example_set'] = st.session_state['fix_answer_set']
        st.write(":green[Perfect, you can get back to the previous tab]")
        st.button("OK")

with tab3: 
    
    ## build the conversation history string and questions string 
    lines = st.session_state.package['questions_str'].splitlines()
    nq = st.session_state.package['questions_num']
    qs = ""
    for n in range(1, nq + 1):    
        qs += (f"- Q{n}: {lines[n-1].strip()}\n \n") 
    

    st.write("This tab just shows you what configuration your bot has right now.")
    st.divider()

    st.markdown(f"**Your questions:** \n")
    st.write(qs)
    st.divider()
    st.markdown(f"**Your one-shot example:** \n")
    st.write(st.session_state.package['current_example_set'])
    st.divider()
    st.markdown(f"**Your current persona:** \n")
    st.write(st.session_state.package['persona'])