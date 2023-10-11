import streamlit as st
from googleapiclient.discovery import build
import smtplib

# Page configuration
st.set_page_config(
    page_title="My Portfolio",
    page_icon=":smiley:",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Page 1", "Page 2"])
if page == "Home":
    st.title("Welcome to my portfolio!")
    st.write("Here you can find information about my projects and more.")
    st.image("Assets/Me.jpeg", width=300)
    st.header("About Me")
    st.write("I'm a software engineer with a passion for machine learning and data science. \nI'm currently pursuing a Bachelor's degree in Computer Science at Indian Institute of Information Technology Kalyani (Graduation 2024).\n")
    st.header("Education")
    st.write("B.Tech in Computer Science and Engineering from Indian Institute of Information Technology, West Bengal, India.I have a CGPA of 8.9/10.")
    st.image("Assets/Screenshot from 2023-10-11 22-23-40.png", width=300)
    st.header("Internships")
    st.subheader("Proficient Vision Solutions Pvt. Ltd. (May 2023 - July 2023), IIT Kharagpur")
    st.write("I worked as a image processing and computer vision intern. I trained SOTA models such as YOLO-NAS and YOLO-V8 on custom data.\nI deployed the models on a Jetson Nano and a Raspberry Pi 4.\nI also used various deraing and dehazing methods to improve the quality of the video stream.")
    st.image("Assets/Screenshot from 2023-10-11 22-33-38.png", width=300)
    st.header("Skills")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Programming Languages")
        st.write("Python")
        st.write("C++")
        st.write("Goland")
        st.write("Java")
    with col2:
        st.subheader("Machine Learning")
        st.write("OpeCV\n")
        st.write("Tensorflow\n")
        st.write("ScikitLearn\n")
        st.write("Keras\n")
        st.write("Pytorch\n")
        st.write("OpenAi API\n")
        st.write("Langchain Agents\n")
        st.image("Assets/st,small,845x845-pad,1000x1000,f8f8f8.jpg", width=300)
    with col3:
        st.subheader("Web Development")
        st.write("Flask\n")
        st.write("Streamlit\n")
        st.write("Django\n")


if page == "Page 1":
    # Header
    st.title("HAL4500: Bridging the Gap Between Sci-Fi and Reality!")

    # Introduction
    st.write("Greetings, fellow space enthusiasts and tech aficionados!")
    st.write("🚀✨ Remember HAL9000 from the iconic '2001: A Space Odyssey'? HAL9000's legacy lives on as we embark on a journey to bring the future closer to the present with HAL4500! 🤖🌠")
    st.write("Imagine a world where machines understand us, collaborate with us, and assist us in real-time.")
    st.write("Well, HAL4500 is here to take us one step closer to that vision. 🌐🔮")

    # Object Detection
    st.subheader("Object Detection Magic")
    st.write("Our journey starts with YOLOv8, a state-of-the-art object detection model trained on the extensive MS COCO dataset.")
    st.write("HAL4500, like a digital detective, can effortlessly detect and recognize objects held in your hands. 📦🔍")

    # LangChain Logic
    st.subheader("LangChain Logic")
    st.write("But HAL4500 doesn't stop there. It's powered by LangChain, a dynamic autonomous agent capable of logic-based decision-making.")
    st.write("HAL4500 can understand your voice commands, engage in conversations, and decide when to deploy its digital tools. It's like having a knowledgeable companion at your fingertips. 💬🤯")

    # Instructions
    st.subheader("Instructions")
    st.markdown("0. Create a `.env` file in the root and add the following lines:")
    st.markdown("```\nHEARING_PORT = ****\nOPEN_AI_API = ********************************\n```")
    st.markdown("1. Install the dependencies using `environment.yaml` or `environment.txt`.")
    st.markdown("2. Run the `vision.py` script.")
    st.markdown("3. Run the `hearing.py` script.")
    st.markdown("4. Run the `main.py` script.")

    st.video("https://user-images.githubusercontent.com/70876392/269930562-655084a3-7521-4d71-b70c-9684385a5c94.mp4")
    # Define function to display link
    def display_link(link_url):
        st.markdown(f'[GITHUB REPO]({link_url})')

    st.markdown("[GitHub Repository](https://github.com/AkashParua/hal4500.git)")



if page == "Page 2":
    # Header
    st.title("TutorMe2 - Project Overview")

    # Introduction
    st.write("This project is improved version of existing project tutome. The prvious version was a Flask application with REST endpoints for semantic search using the same idea")
    st.write("This project demonstrates the integration of multiple NLP models and databases in a Streamlit application.")
    st.write("It uses OpenAI's GPT-3 to answer questions based on provided contexts.")
    st.write("The application utilizes the following libraries:")

    # List of Libraries
    st.write("- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers)")
    st.write("- [Streamlit](https://streamlit.io/)")
    st.write("- [IPython](https://ipython.org/)")
    st.write("- [gTTS](https://gtts.readthedocs.io/en/latest/)")
    st.write("- [audio-recorder-streamlit](https://github.com/dvcrn/audio-recorder-streamlit)")
    st.write("- [SpeechRecognition](https://github.com/Uberi/speech_recognition)")
    st.write("- [PyPDF2](https://github.com/mstamy2/PyPDF2)")
    st.write("- [Pillow](https://python-pillow.org/)")
    st.write("- [dotenv](https://github.com/theskumar/python-dotenv)")
    st.write("- [Pinecone](https://www.pinecone.io/)")
    st.write("- [requests](https://requests.readthedocs.io/en/latest/)")
    st.write("- [pytesseract](https://github.com/madmaze/pytesseract)")

    # Usage Instructions
    st.subheader("Usage Instructions")

    st.write("Before running the application, you will need to create a `.env` file in the root directory of the project with the following keys:")
    st.code("PINECONE_KEY: API key for Pinecone\nGPT_KEY: API key for OpenAI GPT-3", language="plaintext")

    st.write("Once you have created the `.env` file, you can run the application by running the following commands in your terminal:")

    st.code("git clone https://github.com/AkashParua/TutorMe2.git", language="shell")
    st.code("cd TutorMe2", language="shell")
    st.code("pip install -r requirements.txt", language="shell")
    st.code("streamlit run app.py", language="shell")

    st.write("Follow the instructions in the Streamlit app to interact with the different features.")

    # Features
    st.subheader("Features")

    # Text to Speech
    st.write("### Text to Speech")
    st.write("This section allows you to enter text and convert it to speech. The text is converted to an audio file using the **gTTS** library and played back using the **IPython.display** library.")

    # Speech to Text
    st.write("### Speech to Text")
    st.write("This section allows you to record audio and transcribe it to text using the **speech_recognition** library. The audio is recorded using the **audio-recorder-streamlit** library and saved to a **.wav** file using the **wave** library. The text is transcribed using the Google Speech Recognition API.")

    # OCR
    st.write("### OCR")
    st.write("This section allows you to upload an image file and extract text using OCR. The image file is loaded using the **Pillow** library and the text is extracted using the **pytesseract** library.")

    # Question Answering
    st.write("### Question Answering")
    st.write("This section allows you to ask a question and receive an answer from a pre-built corpus of text. The corpus is indexed using the **Pinecone** library and the embeddings are generated using the **SentenceTransformers** library. The question is then answered using the OpenAI GPT-3 API.")

    # Database Update
    st.write("### Database Update")
    st.write("This section allows you to upload a PDF file and update the pre-built corpus of text. The PDF file is loaded using the **PyPDF2** library and the text is extracted from each page. The text is then split into chunks and indexed using the **Pinecone** library.")

    # License
    st.subheader("License")
    st.write("This project is licensed under the MIT License - see the **LICENSE.md** file for details.")

    # Display external links as hyperlinks
    st.markdown("[GitHub Repository](https://github.com/AkashParua/TutorMe2.git)")



# Define your projects
# Footer
st.markdown("---")
st.write("Connect with me on [LinkedIn](https://www.linkedin.com/in/akash-parua-76b2531b7/)")
st.write("Connect with me via : akashparua999@gmail.com")
st.write("Feel free to reach out to me via email or LinkedIn.")
