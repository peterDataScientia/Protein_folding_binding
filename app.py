# This is app is created with instructions provided by Chanin Nantasenamat (Data Professor) https://youtube.com/dataprofessor
# Credit: This app is inspired by https://huggingface.co/spaces/osanseviero/esmfold

import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import altair as alt
from PIL import Image

# st.set_page_config(layout = 'wide')
######################
# Page title
st.markdown("""
# Protein structure prediction using ESMFold

This interactive App enable you to predict protein structure from the amino acid sequence, based on the ESM2 large language model.

""")

image = Image.open('logo.jpg')
st.image(image, use_column_width=True)
st.sidebar.title('ESMFold')
st.sidebar.write('[*ESMFold*](https://esmatlas.com/about) is an end-to-end single sequence protein structure predictor based on the ESM-2 language model. For more information, read the [research article](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2) and the [news article](https://www.nature.com/articles/d41586-022-03539-1) published in *Nature*.')

# stmol
def render_mol(pdb):

    pdbview = py3Dmol.view()
    pdbview.addModel(pdb,'pdb')
    style = st.selectbox('style', ['cartoon','line','cross','stick','sphere'], on_change=update)
    spinon = st.button('Spin ON', on_click=update)
    spinoff = st.button('Spin OFF', on_click=update)
    pdbview.setStyle({style:{'color':'spectrum'}})
    pdbview.setBackgroundColor('white')#('0xeeeeee')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True == spinon, False == spinoff)
    showmol(pdbview, height = 500,width=800)

# Protein sequence input
DEFAULT_SEQ = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
txt = st.sidebar.text_area('Input sequence [upto 400 amino acids]', DEFAULT_SEQ, height=275)

# ESMfold
def update(sequence=txt):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=sequence, verify=False)
    if response.status_code != 200:
        st.warning("Unable to predict protein structure. Please try again later.")
        return

    name = sequence[:3] + sequence[-3:]
    pdb_string = response.content.decode('utf-8')

    with open('predicted.pdb', 'w') as f:
        f.write(pdb_string)

    struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
    b_value = round(struct.b_factor.mean(), 4)

    # Display protein structure
    st.subheader('Visualization of predicted protein structure')
    render_mol(pdb_string)

    # plDDT value is stored in the B-factor field
    st.subheader('pLDDT value')
    st.write('pLDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100. (higher the number, higher the confidance of structure predicted)')
    st.info(f'pLDDT: {b_value * 100}')

    st.download_button(
        label="Download PDB",
        data=pdb_string,
        file_name='predicted.pdb',
        mime='text/plain',
    )

predict = st.sidebar.button('Predict', on_click=update)


if not predict:
    st.warning('👈 Enter protein sequence data in side panel!')

st.markdown("""
***Found the App useful or Have a suggestion?*** Kinldy provide your valuable FEEDBACK [HERE](https://forms.gle/nuSvvnqZ3Aofmb3p7)

*Credits:
This app is inspired by the work done by [Chanin Nantasenamat](https://github.com/dataprofessor) & [osanseviero](https://huggingface.co/spaces/osanseviero/esmfold)*

**Protein structure prediction using ESMFold App** © 2024 by [VISHAL BHOIR](https://linktr.ee/thebioway) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

""")
