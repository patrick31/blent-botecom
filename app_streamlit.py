"""Interface Streamlit optionnelle pour tester l'assistant."""

from __future__ import annotations

import streamlit as st

from customer_service_bot.assistant import CustomerServiceAssistant
from customer_service_bot.config import AppConfig
from customer_service_bot.database import OrderRepository


st.set_page_config(page_title="Assistant service client")
st.title("Assistant service client")

config = AppConfig.from_env()
repository = OrderRepository(config.database_path)


@st.cache_resource
def get_assistant(runtime_config: AppConfig) -> CustomerServiceAssistant:
    """Charge l'assistant une seule fois par configuration Streamlit."""

    return CustomerServiceAssistant(
        config=runtime_config,
        repository=OrderRepository(runtime_config.database_path),
    )

email = st.text_input("Email client", value="pellentesque.ultricies@protonmail.com")
question = st.text_area("Question", value="Ou en est ma commande 4 ?")
st.caption("Le LLM Hugging Face est utilise pour analyser la question et generer la reponse.")

if st.button("Envoyer"):
    user = repository.get_user_by_email(email)
    if user is None:
        st.error("Utilisateur introuvable.")
    else:
        runtime_config = AppConfig(
            database_path=config.database_path,
            llm_model_id=config.llm_model_id,
            embedding_model_id=config.embedding_model_id,
            max_new_tokens=config.max_new_tokens,
            analysis_max_new_tokens=config.analysis_max_new_tokens,
            temperature=config.temperature,
            llm_backend=config.llm_backend,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
            load_in_4bit=config.load_in_4bit,
            use_llm=True,
        )
        try:
            assistant = get_assistant(runtime_config)
            history_key = f"history:{user.email}"
            history = st.session_state.setdefault(history_key, [])
            answer = assistant.answer(user, question, history=history)
            st.write(answer)
            history.append((question, answer))
        except Exception as exc:
            st.error(f"Erreur pendant l'appel au LLM: {exc}")
