import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import biotite.structure as bs
import altair as alt
from PIL import Image
import re
import uuid
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🧬 AI Protein Structure Prediction & Analysis Platform")

# -----------------------------
# LOAD IMAGE
# -----------------------------
try:
    image = Image.open('logo.jpg')
    st.image(image, use_column_width=True)
except:
    st.warning("Logo not found.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("ESMFold Platform")

sequence = st.sidebar.text_area(
    "Input sequence (≤ 400 amino acids)",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    height=250
)

predict = st.sidebar.button("Predict & Analyze")

# -----------------------------
# VALIDATION
# -----------------------------
def is_valid_sequence(seq):
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq))

# -----------------------------
# FETCH STRUCTURE
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_structure(sequence):
    try:
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=sequence,
            timeout=60
        )
        response.raise_for_status()
        return response.content.decode('utf-8')
    except:
        return None

# -----------------------------
# BINDING SITE PREDICTION
# -----------------------------
def predict_binding_sites(struct):
    try:
        coords = struct.coord
        dist_matrix = distance_matrix(coords, coords)
        density = (dist_matrix < 6.0).sum(axis=1)
        top_indices = np.argsort(density)[-20:]
        return top_indices
    except:
        return None

# -----------------------------
# RAMACHANDRAN PLOT
# -----------------------------
def ramachandran_plot(struct):
    try:
        phi, psi, _ = bs.dihedral_backbone(struct)

        mask = ~np.isnan(phi) & ~np.isnan(psi)
        phi = np.degrees(phi[mask])
        psi = np.degrees(psi[mask])

        df = pd.DataFrame({"phi": phi, "psi": psi})

        chart = alt.Chart(df).mark_circle(size=40).encode(
            x=alt.X("phi", scale=alt.Scale(domain=[-180, 180])),
            y=alt.Y("psi", scale=alt.Scale(domain=[-180, 180]))
        ).properties(title="Ramachandran Plot")

        st.altair_chart(chart, use_container_width=True)

    except:
        st.warning("Ramachandran plot failed.")

# -----------------------------
# 3D VISUALIZATION
# -----------------------------
def render_mol(pdb_string, binding_sites=None):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb_string, 'pdb')

    pdbview.setStyle({
        "cartoon": {
            "colorscheme": {
                "prop": "b",
                "gradient": "roygb",
                "min": 0,
                "max": 100
            }
        }
    })

    if binding_sites is not None:
        for resi in binding_sites:
            pdbview.addStyle(
                {"resi": int(resi)},
                {"stick": {"color": "red"}}
            )

    pdbview.setBackgroundColor('white')
    pdbview.zoomTo()
    showmol(pdbview, height=500, width=800)

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if predict:

    sequence = sequence.strip().upper()

    if len(sequence) > 400:
        st.error("Sequence too long! Max = 400 amino acids")
        st.stop()

    if not is_valid_sequence(sequence):
        st.error("Invalid sequence!")
        st.stop()

    with st.spinner("Predicting structure..."):
        pdb_string = fetch_structure(sequence)

    if pdb_string is None:
        st.error("Prediction failed. Try again.")
        st.stop()

    # Save structure
    filename = f"pred_{uuid.uuid4().hex}.pdb"
    with open(filename, "w") as f:
        f.write(pdb_string)

    struct = bsio.load_structure(filename, extra_fields=["b_factor"])
    plddt = struct.b_factor * 100
    avg_plddt = round(plddt.mean(), 2)

    # Binding sites
    binding_sites = predict_binding_sites(struct)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    st.subheader("🧪 3D Structure (Confidence Heatmap + Binding Sites)")
    render_mol(pdb_string, binding_sites)

    # pLDDT
    st.subheader("📊 Confidence Score")
    st.info(f"Average pLDDT: {avg_plddt}")

    # Histogram
    df = pd.DataFrame(plddt, columns=["pLDDT"])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("pLDDT", bin=True),
        y="count()"
    )
    st.altair_chart(chart, use_container_width=True)

    # Ramachandran
    st.subheader("📐 Ramachandran Plot")
    ramachandran_plot(struct)

    # Binding residues
    if binding_sites is not None:
        st.subheader("🎯 Predicted Binding Residues")
        st.write(binding_sites.tolist())

    # Download
    st.download_button(
        "Download PDB",
        pdb_string,
        file_name=filename
    )

else:
    st.info("👈 Enter sequence and click Predict & Analyze")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Integrated AI platform: Prediction + Validation + Interpretation")