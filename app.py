"""A small local website for training, watching, and testing your rocket-landing AI."""

from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

from evaluate import evaluate_model
from train import VALID_MODES, model_path_for, train_agent

st.set_page_config(page_title="Rocket Lander AI Lab", page_icon="🚀", layout="wide")

MODE_LABELS = {
    "standard": "Original reward (baseline)",
    "time_penalty": "Improved experiment (small time penalty)",
    "hover_bonus": "Intentionally flawed (hover bonus)",
}
MODE_EXPLANATIONS = {
    "standard": "The game gives the AI its normal score for moving toward a safe landing.",
    "time_penalty": "The game score gets a tiny −0.02 penalty each frame. This discourages wasting time; it is an experiment, not a guaranteed improvement.",
    "hover_bonus": "The AI gets +1 when it is upright and moving slowly, even when it has not landed. This can teach the wrong behaviour: reward hacking.",
}

st.title("🚀 Rocket Lander AI Lab")
st.write("Train a small neural network to land a rocket in a physics simulator. Change the reward rules and compare what your own AI learns.")
st.caption("Runs on your computer. Your trained models are saved in the project's models folder.")

with st.expander("How the AI learns", expanded=False):
    st.markdown(
        """
        1. **Observe:** the game gives the AI 8 numbers, such as height, sideways speed, tilt, and whether the legs touch the ground.
        2. **Act:** the AI chooses one of 4 engine actions.
        3. **Score:** the environment gives a reward for what happened.
        4. **Learn:** PPO updates a neural network so actions that lead to better results become more likely.
        5. **Repeat:** the AI tries thousands of frames. It may crash at first; that is normal.
        """
    )

st.divider()

train_tab, watch_tab, evaluate_tab, experiment_tab = st.tabs(
    ["1. Train AI", "2. Watch it fly", "3. Test results", "4. Reward experiment"]
)

with train_tab:
    st.subheader("Train a new AI")
    left, right = st.columns([1.2, 1])
    with left:
        selected_mode = st.selectbox(
            "Reward rule",
            options=list(VALID_MODES),
            format_func=lambda mode: MODE_LABELS[mode],
            key="training_reward_mode",
        )
        timesteps = st.select_slider(
            "Training steps (each step is one game frame)",
            options=[10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000],
            value=200_000,
            format_func=lambda n: f"{n:,}",
        )
        st.info(MODE_EXPLANATIONS[selected_mode])
        st.caption("Start with 100,000–200,000 steps to check that everything works. Better landings may need 500,000–1,000,000 steps and more than one run.")
        start_training = st.button("Train AI now", type="primary", use_container_width=True)
    with right:
        st.markdown("**What happens when you train?**")
        st.write("A new PPO neural network is created and learns from simulator experience. It is not a pretrained rocket model.")
        st.write("Training can take from minutes to longer depending on your computer and the selected number of steps. You can use the website while the training is running, but this training run itself will occupy the app until it finishes.")
        st.write("Each reward rule gets its own save file, so you can compare all three versions.")

    if start_training:
        progress = st.progress(0.0, text="Preparing the simulation…")
        status = st.empty()

        def update_progress(current, total, average_reward):
            fraction = min(current / max(total, 1), 1.0)
            progress.progress(fraction, text=f"Training: {current:,} / {total:,} frames")
            if average_reward is None:
                status.write("The AI is practising; average episode score will appear after it completes an episode.")
            else:
                status.write(f"Average shaped reward across recent completed episodes: {average_reward:.1f}")

        try:
            with st.spinner("The AI is learning. Please keep this page open until training finishes…"):
                saved_path = train_agent(
                    total_timesteps=int(timesteps),
                    reward_mode=selected_mode,
                    update_callback=update_progress,
                )
            progress.progress(1.0, text="Training complete")
            st.success(f"Your new AI was saved: {saved_path.relative_to(Path(__file__).resolve().parent)}")
            st.write("Next: open **Watch it fly** to see the model, then use **Test results** to evaluate it with the original game reward.")
        except Exception as exc:
            st.error("Training did not finish. Read the error details below, then check the setup steps in README.md.")
            st.exception(exc)
            st.caption("If the error mentions Box2D or SWIG, the README explains the extra installation step.")

with watch_tab:
    st.subheader("Watch your saved AI")
    mode_to_watch = st.selectbox(
        "Which trained model?",
        options=list(VALID_MODES),
        format_func=lambda mode: MODE_LABELS[mode],
        key="watch_reward_mode",
    )
    demo_frames = st.slider("Maximum frames for this demo", min_value=200, max_value=1000, value=600, step=100)
    model_file = model_path_for(mode_to_watch)
    if model_file.exists():
        st.caption(f"Saved model found: {model_file.relative_to(Path(__file__).resolve().parent)}")
        if st.button("Launch rocket demo", type="primary"):
            picture = st.empty()
            demo_status = st.empty()
            env = gym.make("LunarLander-v3", render_mode="rgb_array")
            try:
                model = PPO.load(str(model_file))
                observation, _ = env.reset(seed=2026)
                total_reward = 0.0
                ended = False
                frame_count = 0
                for frame_count in range(1, demo_frames + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    frame = env.render()
                    if frame_count % 4 == 0 or terminated or truncated:
                        picture.image(frame, channels="RGB", use_container_width=True)
                        demo_status.write(f"Frame {frame_count}/{demo_frames} · Original game score: {total_reward:.1f}")
                        time.sleep(0.02)
                    if terminated or truncated:
                        ended = True
                        break
                if total_reward >= 200:
                    st.success(f"Great result for this trial: score {total_reward:.1f} (200+ is the usual solved threshold).")
                elif ended:
                    st.warning(f"The episode finished with score {total_reward:.1f}. It may need more training; try a longer run.")
                else:
                    st.info(f"Demo stopped after {frame_count} frames. Score so far: {total_reward:.1f}.")
            except Exception as exc:
                st.error("Could not run the demo.")
                st.exception(exc)
            finally:
                env.close()
        with open(model_file, "rb") as model_download:
            st.download_button(
                "Download this trained model (.zip)",
                data=model_download.read(),
                file_name=model_file.name,
                mime="application/zip",
            )
    else:
        st.warning("No model saved for this reward rule yet. Go to **Train AI** and train that version first.")

with evaluate_tab:
    st.subheader("Test whether it has learned to land")
    st.write("Evaluation deliberately uses the game's original reward with all custom reward changes switched off. This prevents a bonus from making a bad landing look successful.")
    eval_mode = st.selectbox(
        "Model to evaluate",
        options=list(VALID_MODES),
        format_func=lambda mode: MODE_LABELS[mode],
        key="evaluation_reward_mode",
    )
    episodes = st.selectbox("Number of test attempts", options=[5, 10, 20], index=1)
    eval_path = model_path_for(eval_mode)
    if eval_path.exists():
        if st.button("Run evaluation", type="primary"):
            try:
                with st.spinner(f"Testing {episodes} attempts with the original game reward…"):
                    result = evaluate_model(eval_path, episodes=int(episodes))
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Average score", f"{result['average_score']:.1f}")
                metric2.metric("Scores of 200+", f"{result['successes']}/{result['episodes']}")
                metric3.metric("Success rate", f"{result['success_rate'] * 100:.0f}%")
                score_df = pd.DataFrame({"Attempt": list(range(1, episodes + 1)), "Original game score": result["scores"]}).set_index("Attempt")
                st.line_chart(score_df)
                st.dataframe(score_df, use_container_width=True)
                st.caption("A single run can be lucky or unlucky. Compare several attempts and, ideally, repeat training with another seed before making a strong claim.")
            except Exception as exc:
                st.error("Evaluation failed.")
                st.exception(exc)
    else:
        st.warning("Train this version first, then run the evaluation.")

with experiment_tab:
    st.subheader("Experiment: can a reward teach the wrong thing?")
    st.markdown(
        """
        The project is about **reward shaping**: changing points to guide the AI's behaviour.

        | Version | Reward change | What to look for |
        |---|---|---|
        | Original reward | No extra rule | Your baseline landing performance |
        | Small time penalty | −0.02 each frame | Whether it finishes sooner without harming landing quality |
        | Hover bonus (deliberately bad) | +1 when upright and slow | Whether it learns to seek the bonus instead of landing |

        **Reward hacking** happens when an AI finds a way to earn the points you gave it without actually completing the goal. It is not guaranteed that every run will visibly hover; training is random. Check the video and the original-reward evaluation, not just the training score.
        """
    )
    st.write("Suggested mini-experiment: train each version for the same number of steps (for example, 300,000), run 10 evaluation attempts per version, and compare the average scores and videos.")
    st.write("The anti-hover lesson is to avoid giving a continuing bonus just for being in a comfortable state. Reward the real task, and test models using the original game reward.")

st.divider()
st.caption("Built with Gymnasium LunarLander-v3, Stable-Baselines3 PPO, and Streamlit. See README.md for install instructions and command-line alternatives.")
