import io
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from logic import (
    generate_images,
    inpaint_image,
    outpaint_image,
    clear_memory,
)

st.set_page_config(
    page_title="Generative Image Studio - BFGAI",
    page_icon="🎨",
    layout="wide"
)

if "generated_images" not in st.session_state:
    st.session_state["generated_images"] = []
if "active_image" not in st.session_state:
    st.session_state["active_image"] = None

with st.sidebar:
    st.header("⚙️ System & Memory")
    st.caption("Model: Stable Diffusion v1.5")
    if st.button("🧹 Clear Memory", use_container_width=True):
        clear_memory()
        st.success("Memori RAM & VRAM GPU berhasil dibersihkan!")

    st.divider()
    if st.session_state["active_image"] is not None:
        st.subheader("🖼️ Active Image for Edit")
        st.image(st.session_state["active_image"], use_column_width=True)
    else:
        st.info("Generate gambar terlebih dahulu pada Tab Text-to-Image untuk langsung diedit di Tab Inpainting & Outpainting.")

st.title("🎨 Generative Image Suite (Stable Diffusion 1.5)")

tab_gen, tab_inpaint, tab_outpaint = st.tabs([
    "✨ Text-to-Image Generation",
    "🖌️ Inpainting (Canvas Mask)",
    "🔭 Outpainting & Zoom Out"
])

# ==========================================
# TAB 1: TEXT-TO-IMAGE GENERATION
# ==========================================
with tab_gen:
    col_ctrl, col_out = st.columns([1, 1.3], gap="large")

    with col_ctrl:
        st.subheader("Parameter Generasi")
        prompt = st.text_area(
            "Prompt",
            value="2d digital art illustration of an astronaut in a white spacesuit standing on the moon surface, planet earth in starry sky, clean vector art",
            height=110
        )
        negative_prompt = st.text_area(
            "Negative Prompt",
            value="photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration",
            height=90
        )

        c1, c2 = st.columns(2)
        with c1:
            scheduler_name = st.selectbox(
                "Scheduler",
                options=["Euler A", "DPM++", "DDIM"],
                index=1
            )
            guidance_scale = st.slider(
                "Guidance Scale (CFG)",
                min_value=1.0,
                max_value=20.0,
                value=8.5,
                step=0.5
            )
        with c2:
            num_images = st.selectbox(
                "Jumlah Gambar (Grid 2x2)",
                options=[1, 2, 4],
                index=2
            )
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=10,
                max_value=50,
                value=30,
                step=5
            )

        seed = st.number_input("Seed", min_value=0, max_value=999999, value=222, step=1)
        btn_generate = st.button("🚀 Generate Image", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil Generasi")
        if btn_generate:
            if not prompt.strip():
                st.warning("Masukkan prompt terlebih dahulu.")
            else:
                with st.spinner("Sedang membuat gambar dengan Stable Diffusion v1.5..."):
                    imgs = generate_images(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_images=num_images,
                        scheduler_name=scheduler_name,
                        seed=seed
                    )
                    st.session_state["generated_images"] = imgs
                    st.session_state["active_image"] = imgs[0]

        if st.session_state["generated_images"]:
            imgs = st.session_state["generated_images"]
            if len(imgs) == 1:
                st.image(imgs[0], caption=f"Generated (Seed={seed})", use_column_width=True)
            else:
                for row_start in range(0, len(imgs), 2):
                    grid_cols = st.columns(2)
                    for idx_offset in range(2):
                        img_idx = row_start + idx_offset
                        if img_idx < len(imgs):
                            with grid_cols[idx_offset]:
                                st.image(
                                    imgs[img_idx],
                                    caption=f"Image #{img_idx + 1} (Seed={int(seed) + img_idx})",
                                    use_column_width=True
                                )
                                if st.button(f"Pilih Gambar #{img_idx + 1} untuk Edit", key=f"select_img_{img_idx}"):
                                    st.session_state["active_image"] = imgs[img_idx]
                                    st.rerun()

# ==========================================
# TAB 2: INPAINTING DENGAN DRAWABLE CANVAS
# ==========================================
with tab_inpaint:
    st.subheader("🖌️ Inpainting dengan Masking Interaktif")
    if st.session_state["active_image"] is None:
        st.warning("Belum ada gambar aktif. Silakan generate gambar di Tab Text-to-Image terlebih dahulu.")
    else:
        base_img = st.session_state["active_image"].resize((512, 512))
        col_canvas, col_inp_ctrl = st.columns([1.1, 1], gap="large")

        with col_canvas:
            st.markdown("**Gambar Mask pada Area yang Ingin Diubah:**")
            stroke_width = st.slider("Ukuran Kuas (Brush Size)", min_value=5, max_value=60, value=25)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.5)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_image=base_img,
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode="freedraw",
                key="inpaint_canvas"
            )

        with col_inp_ctrl:
            inpaint_prompt = st.text_area(
                "Prompt Inpainting",
                value="2d digital art illustration of a broken crashed satellite with solar panels on the moon, clean vector art",
                height=100
            )
            inpaint_neg = st.text_area(
                "Negative Prompt Inpainting",
                value="photorealistic, blurry, low quality, messy, ugly, watermark",
                height=80
            )
            inpaint_strength = st.slider("Strength", min_value=0.3, max_value=1.0, value=0.95, step=0.05)
            inpaint_cfg = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=9.0, step=0.5, key="inp_cfg")
            inpaint_steps = st.slider("Inference Steps", min_value=15, max_value=50, value=30, step=5, key="inp_steps")
            inpaint_seed = st.number_input("Seed Inpainting", min_value=0, max_value=999999, value=9, step=1)

            btn_inpaint = st.button("✨ Jalankan Inpainting", type="primary", use_container_width=True)

            if btn_inpaint:
                if canvas_result.image_data is None or np.max(canvas_result.image_data[:, :, 3]) == 0:
                    st.warning("Silakan coret area mask pada gambar di sebelah kiri terlebih dahulu.")
                else:
                    alpha_channel = canvas_result.image_data[:, :, 3]
                    mask_array = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_array, mode="L")

                    with st.spinner("Memproses Inpainting..."):
                        inpainted_res = inpaint_image(
                            image=base_img,
                            mask=mask_pil,
                            prompt=inpaint_prompt,
                            negative_prompt=inpaint_neg,
                            strength=inpaint_strength,
                            guidance_scale=inpaint_cfg,
                            num_inference_steps=inpaint_steps,
                            seed=inpaint_seed
                        )
                        st.session_state["inpainted_result"] = inpainted_res
                        st.session_state["inpainted_mask"] = mask_pil

            if "inpainted_result" in st.session_state:
                st.divider()
                r1, r2 = st.columns(2)
                with r1:
                    st.image(st.session_state["inpainted_mask"], caption="Mask", use_column_width=True)
                with r2:
                    st.image(st.session_state["inpainted_result"], caption="Hasil Inpainting", use_column_width=True)
                if st.button("📌 Gunakan Hasil Ini untuk Outpainting", use_container_width=True):
                    st.session_state["active_image"] = st.session_state["inpainted_result"]
                    st.rerun()

# ==========================================
# TAB 3: OUTPAINTING & ZOOM OUT
# ==========================================
with tab_outpaint:
    st.subheader("🔭 Outpainting & Zoom Out")
    if st.session_state["active_image"] is None:
        st.warning("Belum ada gambar aktif. Silakan generate gambar di Tab Text-to-Image terlebih dahulu.")
    else:
        col_out_ctrl, col_out_res = st.columns([1, 1.3], gap="large")

        with col_out_ctrl:
            outpaint_mode = st.selectbox(
                "Mode Perluasan Kanvas",
                options=["Zoom Out (All Directions)", "One-Sided Outpainting"]
            )
            direction = "Right"
            if outpaint_mode == "One-Sided Outpainting":
                direction = st.selectbox("Arah Perluasan", options=["Right", "Left", "Top", "Bottom"])

            expand_px = st.slider("Ukuran Perluasan (Pixel)", min_value=64, max_value=256, value=128, step=64)
            outpaint_prompt = st.text_area(
                "Prompt Outpainting / Zoom Out",
                value="wide angle 2d digital art illustration of an astronaut on the moon surface, vast lunar craters, planet earth and stars in dark sky, clean vector art",
                height=100
            )
            outpaint_neg = st.text_area(
                "Negative Prompt",
                value="photorealistic, blurry, low quality, border, frame, cut off, messy",
                height=80,
                key="out_neg"
            )
            out_cfg = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=8.5, step=0.5, key="out_cfg")
            out_steps = st.slider("Inference Steps", min_value=15, max_value=50, value=30, step=5, key="out_steps")
            out_seed = st.number_input("Seed Outpainting", min_value=0, max_value=999999, value=222, step=1)

            btn_outpaint = st.button("🌌 Jalankan Outpainting / Zoom Out", type="primary", use_container_width=True)

        with col_out_res:
            if btn_outpaint:
                with st.spinner("Memperluas kanvas dan merender Outpainting..."):
                    out_res, out_mask = outpaint_image(
                        image=st.session_state["active_image"],
                        prompt=outpaint_prompt,
                        negative_prompt=outpaint_neg,
                        mode=outpaint_mode,
                        direction=direction,
                        expand_pixels=expand_px,
                        strength=0.95,
                        guidance_scale=out_cfg,
                        num_inference_steps=out_steps,
                        seed=out_seed
                    )
                    st.session_state["outpainted_result"] = out_res
                    st.session_state["outpainted_mask"] = out_mask

            if "outpainted_result" in st.session_state:
                m_col, res_col = st.columns([1, 2])
                with m_col:
                    st.image(st.session_state["outpainted_mask"], caption="Area Mask", use_column_width=True)
                with res_col:
                    st.image(
                        st.session_state["outpainted_result"],
                        caption=f"Hasil {outpaint_mode} ({st.session_state['outpainted_result'].size})",
                        use_column_width=True
                    )
                buf = io.BytesIO()
                st.session_state["outpainted_result"].save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Hasil Gambar (PNG)",
                    data=buf.getvalue(),
                    file_name="hasil_generative_studio.png",
                    mime="image/png",
                    use_container_width=True
                )
