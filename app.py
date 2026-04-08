import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import biotite.structure as bs
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import re
import uuid
import io
import zipfile
import streamlit.components.v1 as components
from io import StringIO

# -------------------- Page Setup --------------------
st.set_page_config(layout="wide")
st.title("🧬 AI Protein Structure Prediction & Analysis (Publication-Ready)")

# Logo
try:
    image = Image.open("logo.jpg")
    st.image(image, width=300)
except:
    st.warning("Logo not found")

# Sidebar
st.sidebar.title("Input Sequence")
sequence = st.sidebar.text_area("Protein sequence (≤400 aa)", height=250)
st.sidebar.write(f"Length: {len(sequence)} aa")
predict = st.sidebar.button("Predict & Analyze")

# -------------------- Helper Functions --------------------
def is_valid_sequence(seq):
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq))

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_structure(sequence):
    try:
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=sequence,
            timeout=60
        )
        response.raise_for_status()
        return response.content.decode("utf-8")
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except:
        return None

def predict_best_pocket(struct):
    coords = struct.coord
    dist_matrix = distance_matrix(coords, coords)
    density = (dist_matrix < 6.0).sum(axis=1)
    threshold = np.percentile(density, 90)
    top_indices = np.where(density >= threshold)[0]
    pocket_coords = coords[top_indices]
    centroid = pocket_coords.mean(axis=0)
    score = density[top_indices].mean()
    return top_indices, centroid, score

def ramachandran_plot(struct):
    phi, psi, _ = bs.dihedral_backbone(struct)
    mask = ~np.isnan(phi) & ~np.isnan(psi)
    phi = np.degrees(phi[mask])
    psi = np.degrees(psi[mask])

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_facecolor("#f7f7f7")
    ax.scatter(phi, psi, alpha=0.6, edgecolor='k')
    ax.set_xlim([-180,180]); ax.set_ylim([-180,180])
    ax.set_xlabel("Phi (°)")
    ax.set_ylabel("Psi (°)")
    ax.set_title("Ramachandran Plot")
    ax.grid(True)
    return fig

def plddt_plot(plddt):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(plddt, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel("pLDDT")
    ax.set_ylabel("Residue Count")
    ax.set_title("pLDDT Confidence Score")
    ax.grid(True)
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', transparent=True)
    buf.seek(0)
    return buf

def create_fig_zip(plddt_fig, ramach_fig, pdb_string, binding_sites=None, centroid=None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("pLDDT.png", fig_to_bytes(plddt_fig).read())
        zf.writestr("Ramachandran.png", fig_to_bytes(ramach_fig).read())
        zf.writestr("Protein_structure.pdb", pdb_string)

        if binding_sites is not None and centroid is not None:
            df = pd.DataFrame({
                "Binding Residue Index": binding_sites.tolist(),
                "Centroid_X": [centroid[0]]*len(binding_sites),
                "Centroid_Y": [centroid[1]]*len(binding_sites),
                "Centroid_Z": [centroid[2]]*len(binding_sites)
            })
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            zf.writestr("BindingPocket.csv", csv_buf.getvalue())

    zip_buffer.seek(0)
    return zip_buffer

def render_mol_with_snapshot(pdb_string, binding_sites=None, centroid=None):
    pdbview_js = f"""
    <div id="container" style="width:800px; height:500px;"></div>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <script>
        var element = document.getElementById("container");
        var viewer = $3Dmol.createViewer(element, {{backgroundColor:"white"}});
        viewer.addModel(`{pdb_string}`, "pdb");
        viewer.setStyle({{cartoon:{{colorscheme:{{prop:"b", gradient:"roygb", min:0, max:100}}}}}});
        {"".join([f'viewer.addStyle({{"resi":{resi}}},{{stick:{{color:"red"}}}});' for resi in binding_sites]) if binding_sites is not None else ""}
        {f'viewer.addSphere({{center:{{x:{centroid[0]},y:{centroid[1]},z:{centroid[2]}}}, radius:2.5, color:"yellow"}});' if centroid is not None else ""}
        viewer.zoomTo();
        viewer.render();

        var btn = document.createElement("button");
        btn.innerHTML = "Download 3D PNG";
        btn.style.marginTop="5px";
        btn.onclick = function(){{
            var imgData = viewer.pngURI({{antialias:true, factor:3}});
            var link = document.createElement('a');
            link.download = 'protein_structure.png';
            link.href = imgData;
            link.click();
        }};
        element.appendChild(btn);
    </script>
    """
    components.html(pdbview_js, height=550, width=820)

# -------------------- Main App --------------------
if predict:
    sequence = sequence.strip().upper()

    # FASTA support
    if sequence.startswith(">"):
        sequence = "".join(sequence.split("\n")[1:])

    if not sequence or len(sequence) > 400 or not is_valid_sequence(sequence):
        st.error("Invalid sequence! Max 400 amino acids, only 20 standard residues allowed.")
        st.stop()

    with st.spinner("Predicting structure..."):
        pdb_string = fetch_structure(sequence)

    if pdb_string == "TIMEOUT":
        st.error("ESMFold API timeout. Try again.")
        st.stop()
    if pdb_string is None:
        st.error("Prediction failed.")
        st.stop()

    # Load structure from memory
    struct = bsio.load_structure(StringIO(pdb_string), extra_fields=["b_factor"])

    # Correct pLDDT scaling
    plddt = struct.b_factor
    avg_plddt = round(plddt.mean(), 2)

    binding_sites, centroid, score = predict_best_pocket(struct)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Length", len(sequence))
    col2.metric("Avg pLDDT", avg_plddt)
    col3.metric("Pocket Score", round(score,2))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["3D Structure","pLDDT","Ramachandran","Binding Pocket"])

    with tab1:
        st.subheader("3D Structure")
        render_mol_with_snapshot(pdb_string, binding_sites, centroid)
        st.download_button("Download PDB", pdb_string, file_name="protein.pdb")

    with tab2:
        plddt_fig = plddt_plot(plddt)
        st.pyplot(plddt_fig)
        st.download_button("Download PNG", fig_to_bytes(plddt_fig), file_name="plddt.png")

    with tab3:
        ramach_fig = ramachandran_plot(struct)
        st.pyplot(ramach_fig)
        st.download_button("Download PNG", fig_to_bytes(ramach_fig), file_name="ramachandran.png")

    with tab4:
        st.write(f"Pocket Score: {round(score,2)}")
        st.write(f"Centroid: X={centroid[0]:.2f}, Y={centroid[1]:.2f}, Z={centroid[2]:.2f}")
        st.write(f"Binding Residues: {binding_sites.tolist()}")

    # ZIP
    with st.expander("Download All Data"):
        zip_bytes = create_fig_zip(plddt_fig, ramach_fig, pdb_string, binding_sites, centroid)
        st.download_button("Download ZIP", zip_bytes, file_name="Protein_Analysis.zip")