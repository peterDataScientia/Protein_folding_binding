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
st.title("🧬 AI Protein Structure Prediction & Structural Analysis Platform")

# -----------------------------
# LOAD IMAGE
# -----------------------------
try:
    image = Image.open('logo.jpg')
    st.image(image, width="stretch")
except:
    st.warning("Logo not found.")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.title("ESMFold Platform")

sequence = st.sidebar.text_area(
    "Input sequence (≤ 400 amino acids)",
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
# BEST POCKET DETECTION
# -----------------------------
def predict_best_pocket(struct):
    coords = struct.coord

    dist_matrix = distance_matrix(coords, coords)
    density = (dist_matrix < 6.0).sum(axis=1)

    top_indices = np.argsort(density)[-30:]
    pocket_coords = coords[top_indices]

    centroid = pocket_coords.mean(axis=0)
    score = density[top_indices].mean()

    return top_indices, centroid, score

# -----------------------------
# RAMACHANDRAN PLOT (PUBLICATION GRADE)
# -----------------------------
def ramachandran_plot(struct):
    phi, psi, _ = bs.dihedral_backbone(struct)

    mask = ~np.isnan(phi) & ~np.isnan(psi)
    phi = np.degrees(phi[mask])
    psi = np.degrees(psi[mask])

    df = pd.DataFrame({"phi": phi, "psi": psi})

    chart = alt.Chart(df).mark_circle(
        size=60,
        opacity=0.6
    ).encode(
        x=alt.X("phi", scale=alt.Scale(domain=[-180, 180]), title="Phi (°)"),
        y=alt.Y("psi", scale=alt.Scale(domain=[-180, 180]), title="Psi (°)"),
        tooltip=["phi", "psi"]
    ).properties(
        title="Ramachandran Plot",
        width=600,
        height=600
    )

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# 3D VISUALIZATION (FIXED COLOR + POCKET)
# -----------------------------
def render_mol(pdb_string, binding_sites=None, centroid=None):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb_string, 'pdb')

    # ✅ Correct pLDDT coloring (0–1 scale)
    pdbview.setStyle({
        "cartoon": {
            "colorscheme": {
                "prop": "b",
                "gradient": "roygb",
                "min": 0,
                "max": 1
            }
        }
    })

    # Highlight binding residues
    if binding_sites is not None:
        for resi in binding_sites:
            pdbview.addStyle(
                {"resi": int(resi)},
                {"stick": {"color": "red"}}
            )

    # Add pocket center sphere
    if centroid is not None:
        pdbview.addSphere({
            "center": {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2])
            },
            "radius": 2.5,
            "color": "yellow"
        })

    pdbview.setBackgroundColor('white')
    pdbview.zoomTo()

    showmol(pdbview, height=500, width=800)

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if predict:

    sequence = sequence.strip().upper()

    if len(sequence) == 0:
        st.error("Please enter a sequence.")
        st.stop()

    if len(sequence) > 400:
        st.error("Sequence too long! Max = 400 amino acids")
        st.stop()

    if not is_valid_sequence(sequence):
        st.error("Invalid sequence!")
        st.stop()

    with st.spinner("Predicting structure..."):
        pdb_string = fetch_structure(sequence)

    if pdb_string is None:
        st.error("Prediction failed.")
        st.stop()

    filename = f"pred_{uuid.uuid4().hex}.pdb"
    with open(filename, "w") as f:
        f.write(pdb_string)

    struct = bsio.load_structure(filename, extra_fields=["b_factor"])
    plddt = struct.b_factor * 100
    avg_plddt = round(plddt.mean(), 2)

    # Pocket prediction
    binding_sites, centroid, score = predict_best_pocket(struct)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    st.subheader("🧪 3D Structure (pLDDT Heatmap + Binding Pocket)")
    render_mol(pdb_string, binding_sites, centroid)

    # pLDDT
    st.subheader("📊 Confidence Score")
    st.info(f"Average pLDDT: {avg_plddt}")

    df = pd.DataFrame(plddt, columns=["pLDDT"])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("pLDDT", bin=True),
        y="count()"
    )
    st.altair_chart(chart, use_container_width=True)

    # Ramachandran
    st.subheader("📐 Ramachandran Plot")
    ramachandran_plot(struct)

    # Pocket info
    st.subheader("🎯 Best Binding Pocket")

    st.write(f"**Pocket Score:** {round(score, 2)}")

    st.write("**Centroid Coordinates (Å):**")
    st.write({
        "X": round(centroid[0], 2),
        "Y": round(centroid[1], 2),
        "Z": round(centroid[2], 2)
    })

    st.write("**Binding Residues (indices):**")
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
st.markdown("🚀 Integrated AI Platform: Prediction + Validation + Functional Analysis")