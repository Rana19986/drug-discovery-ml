
# ============================================================
# Drug Discovery ML Pipeline
# FastAPI + RDKit + XGBoost
# ============================================================

import os
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pydantic import BaseModel, Field

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "drug_solubility_xgb.pkl"
CONFIG_PATH = "model_config.pkl"

N_BITS = 2048
RADIUS = 2
EXPECTED_FEATURES = 2056


# ============================================================
# Load Model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

# Configuration is optional
if os.path.exists(CONFIG_PATH):
    model_config = joblib.load(CONFIG_PATH)
else:
    model_config = {
        "radius": RADIUS,
        "n_bits": N_BITS,
        "feature_count": EXPECTED_FEATURES
    }


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Drug Discovery ML API",
    description=(
        "Machine learning API for molecular solubility prediction, "
        "drug-likeness analysis, and molecular similarity."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request Models
# ============================================================

class MoleculeRequest(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        description="Molecule SMILES string"
    )


class ScreeningMolecule(BaseModel):
    molecule_id: str
    smiles: str


class ScreeningRequest(BaseModel):
    molecules: list[ScreeningMolecule]


class SimilarityMolecule(BaseModel):
    molecule_id: str
    smiles: str


class SimilarityRequest(BaseModel):
    query_smiles: str
    molecules: list[SimilarityMolecule]
    top_n: int = Field(
        default=5,
        ge=1,
        le=100
    )


# ============================================================
# Molecule Validation
# ============================================================

def get_molecule(smiles: str):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            "Invalid SMILES string."
        )

    return mol


# ============================================================
# Molecular Descriptors
# ============================================================

def calculate_descriptors(mol):

    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        mol.GetNumAtoms(),
        Lipinski.RingCount(mol)
    ]


# ============================================================
# Morgan Fingerprint
# ============================================================

def calculate_fingerprint(mol):

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    return np.asarray(
        fingerprint,
        dtype=np.float32
    )


# ============================================================
# Feature Generation
# ============================================================

def prepare_features(smiles: str):

    mol = get_molecule(smiles)

    descriptors = calculate_descriptors(
        mol
    )

    fingerprint = calculate_fingerprint(
        mol
    )

    features = np.concatenate(
        [
            np.asarray(
                descriptors,
                dtype=np.float32
            ),
            fingerprint
        ]
    )

    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Feature mismatch. Expected "
            f"{EXPECTED_FEATURES}, got {len(features)}."
        )

    return features.reshape(
        1,
        -1
    ), mol


# ============================================================
# Lipinski Analysis
# ============================================================

def lipinski_analysis(mol):

    molecular_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = 0
    violation_list = []

    if molecular_weight > 500:
        violations += 1
        violation_list.append(
            "Molecular weight > 500"
        )

    if logp > 5:
        violations += 1
        violation_list.append(
            "LogP > 5"
        )

    if hbd > 5:
        violations += 1
        violation_list.append(
            "H-bond donors > 5"
        )

    if hba > 10:
        violations += 1
        violation_list.append(
            "H-bond acceptors > 10"
        )

    return {
        "violations": violations,
        "drug_like": violations <= 1,
        "violation_details": violation_list
    }


# ============================================================
# Molecular Properties
# ============================================================

def molecular_properties(mol):

    return {
        "molecular_weight": round(
            Descriptors.MolWt(mol),
            4
        ),
        "logp": round(
            Descriptors.MolLogP(mol),
            4
        ),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "tpsa": round(
            Descriptors.TPSA(mol),
            4
        ),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "total_atoms": mol.GetNumAtoms(),
        "rings": Lipinski.RingCount(mol)
    }


# ============================================================
# Solubility Prediction
# ============================================================

def predict_logS(smiles):

    features, mol = prepare_features(
        smiles
    )

    prediction = model.predict(
        features
    )[0]

    return float(prediction), mol


# ============================================================
# Tanimoto Similarity
# ============================================================

from rdkit import DataStructs


def molecular_similarity(
    query_smiles,
    candidate_smiles
):

    query_mol = get_molecule(
        query_smiles
    )

    candidate_mol = get_molecule(
        candidate_smiles
    )

    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        query_mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(
        candidate_mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    return float(
        DataStructs.TanimotoSimilarity(
            query_fp,
            candidate_fp
        )
    )


# ============================================================
# Root Endpoint
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>Drug Discovery ML</title>

        <style>

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #0b1020;
                color: white;
            }

            .container {
                width: 92%;
                max-width: 850px;
                margin: auto;
                padding: 30px 0;
            }

            .card {
                background: #151b2e;
                border: 1px solid #303952;
                border-radius: 20px;
                padding: 25px;
                margin-top: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,.25);
            }

            h1 {
                font-size: 30px;
                margin-bottom: 8px;
            }

            p {
                color: #aeb7cc;
                line-height: 1.6;
            }

            textarea {
                width: 100%;
                min-height: 90px;
                background: #0d1324;
                color: white;
                border: 1px solid #46506b;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                resize: vertical;
                outline: none;
            }

            textarea:focus {
                border-color: #7c8cff;
            }

            button {
                width: 100%;
                margin-top: 15px;
                padding: 14px;
                border-radius: 12px;
                border: 1px solid #7c8cff;
                background: transparent;
                color: white;
                font-size: 16px;
                cursor: pointer;
                transition: .2s;
            }

            button:hover {
                background: #7c8cff;
                transform: translateY(-2px);
            }

            button:active {
                transform: scale(.97);
            }

            .result {
                margin-top: 20px;
                padding: 15px;
                background: #0d1324;
                border-radius: 12px;
                overflow-x: auto;
            }

            .score {
                font-size: 28px;
                font-weight: bold;
            }

            pre {
                white-space: pre-wrap;
                word-break: break-word;
                color: #cbd5e1;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>🧬 Drug Discovery ML</h1>

            <p>
                Molecular solubility prediction and
                drug-likeness analysis powered by
                RDKit and XGBoost.
            </p>

            <div class="card">

                <h2>Analyze Molecule</h2>

                <textarea
                    id="smiles"
                    placeholder="Enter SMILES, e.g. CCO">
                </textarea>

                <button onclick="analyze()">
                    Analyze Molecule
                </button>

                <div
                    id="result"
                    class="result"
                    style="display:none;">
                </div>

            </div>

        </div>

        <script>

            async function analyze() {

                const smiles =
                    document.getElementById(
                        "smiles"
                    ).value.trim();

                const result =
                    document.getElementById(
                        "result"
                    );

                if (!smiles) {

                    result.style.display = "block";

                    result.innerHTML =
                        "Please enter a SMILES string.";

                    return;
                }

                result.style.display = "block";

                result.innerHTML =
                    "⏳ Analyzing molecule...";

                try {

                    const response =
                        await fetch(
                            "/analyze",
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body: JSON.stringify({
                                    smiles: smiles
                                })
                            }
                        );

                    const data =
                        await response.json();

                    if (!response.ok) {

                        throw new Error(
                            data.detail ||
                            "Request failed."
                        );

                    }

                    result.innerHTML = `

                        <div class="score">
                            Predicted LogS:
                            ${data.predicted_logS}
                        </div>

                        <br>

                        <strong>
                            Drug-like:
                        </strong>
                        ${data.lipinski.drug_like}

                        <br><br>

                        <strong>
                            Molecular Weight:
                        </strong>
                        ${data.properties.molecular_weight}

                        <br>

                        <strong>
                            LogP:
                        </strong>
                        ${data.properties.logp}

                        <br>

                        <strong>
                            TPSA:
                        </strong>
                        ${data.properties.tpsa}

                        <br><br>

                        <pre>
${JSON.stringify(data, null, 2)}
                        </pre>

                    `;

                } catch (error) {

                    result.innerHTML =
                        "❌ " + error.message;

                }

            }

        </script>

    </body>
    </html>
    """


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost",
        "model_file": MODEL_PATH,
        "features": EXPECTED_FEATURES,
        "fingerprint": "Morgan",
        "radius": RADIUS,
        "n_bits": N_BITS
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(
    request: MoleculeRequest
):

    try:

        prediction, _ = predict_logS(
            request.smiles
        )

        return {
            "success": True,
            "smiles": request.smiles,
            "predicted_logS": round(
                prediction,
                4
            )
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Complete Molecule Analysis
# ============================================================

@app.post("/analyze")
def analyze(
    request: MoleculeRequest
):

    try:

        prediction, mol = predict_logS(
            request.smiles
        )

        return {
            "success": True,

            "smiles": request.smiles,

            "predicted_logS": round(
                prediction,
                4
            ),

            "properties":
                molecular_properties(mol),

            "lipinski":
                lipinski_analysis(mol)
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Virtual Screening
# ============================================================

@app.post("/screen")
def screen(
    request: ScreeningRequest
):

    if not request.molecules:

        raise HTTPException(
            status_code=400,
            detail="No molecules provided."
        )

    results = []

    for molecule in request.molecules:

        try:

            prediction, mol = predict_logS(
                molecule.smiles
            )

            results.append({

                "molecule_id":
                    molecule.molecule_id,

                "smiles":
                    molecule.smiles,

                "predicted_logS":
                    round(
                        prediction,
                        4
                    ),

                "molecular_weight":
                    round(
                        Descriptors.MolWt(mol),
                        4
                    ),

                "logp":
                    round(
                        Descriptors.MolLogP(mol),
                        4
                    ),

                "tpsa":
                    round(
                        Descriptors.TPSA(mol),
                        4
                    ),

                "drug_like":
                    lipinski_analysis(
                        mol
                    )["drug_like"]

            })

        except Exception as e:

            results.append({

                "molecule_id":
                    molecule.molecule_id,

                "smiles":
                    molecule.smiles,

                "error":
                    str(e)

            })

    # Higher LogS =
