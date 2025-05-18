from flask import Flask, request, jsonify
import joblib
import scipy.sparse
import os

app = Flask(__name__)

# Load the trained model and vectorizers
model = joblib.load("svm_model.pkl")
gene_vectorizer = joblib.load("gene_vector.pkl")
variation_vectorizer = joblib.load("variation_vector.pkl")
text_vectorizer = joblib.load("text_vector.pkl")

# Mapping from mutation class to treatment protocol and links
treatment_map = {
    1: {
        "text": "Standard therapy; consider immunotherapy or PARP inhibitors if BRCA-mutated.",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7405942/"
    },
    2: {
        "text": "Targeted therapy (e.g., kinase inhibitors); if unavailable, use conventional therapy.",
        "url": "https://www.cancer.gov/about-cancer/treatment/types/targeted-therapies"
    },
    3: {
        "text": "Neutral mutation; follow standard treatment protocols.",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4640069/"
    },
    4: {
        "text": "Confirmed Loss-of-Function; standard care or synthetic lethality strategies.",
        "url": "https://www.nature.com/articles/nrc.2016.128"
    },
    5: {
        "text": "Likely neutral; treat based on tumor type and stage.",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8806427/"
    },
    6: {
        "text": "Uncertain significance; do not change treatment, monitor patient.",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6947910/"
    },
    7: {
        "text": "Confirmed Gain-of-Function; apply targeted inhibitors if available.",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6231376/"
    },
    8: {
        "text": "Likely Switch-of-Function; consider experimental or standard therapy.",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6950540/"
    },
    9: {
        "text": "Confirmed Switch-of-Function; use available neomorphic-targeted drugs.",
        "url": "https://www.frontiersin.org/articles/10.3389/fgene.2022.909402/full"
    }
}

@app.route('/', methods=['GET'])
def home():
    """Homepage with project information and API documentation"""
    return jsonify({
        "project_name": "Care Insight",
        "description": "An API for cancer mutation classification and personalized treatment recommendations",
        "creator": "Youssef Ramdan",
        "version": "1.0.0",
        "endpoints": {
            "/": "This documentation page",
            "/predict": "POST - Predicts mutation class and recommends treatment",
            "/health": "GET - Health check for monitoring"
        },
        "usage": {
            "method": "POST to /predict",
            "content_type": "application/json",
            "parameters": {
                "input1": "Gene name (e.g., BRCA1, TP53)",
                "input2": "Genetic variation (e.g., V600E, G12D)",
                "input3": "Clinical text description"
            },
            "example_request": {
                "input1": "BRCA1",
                "input2": "V600E",
                "input3": "Patient has family history of breast cancer"
            },
            "example_response": {
                "prediction": 1,
                "treatment": "Standard therapy; consider immunotherapy or PARP inhibitors if BRCA-mutated.",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7405942/"
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    gene = request.json.get("gene")
    variation = request.json.get("variation")
    text = request.json.get("text")

    # Transform inputs
    gene_vector = gene_vectorizer.transform([gene])
    variation_vector = variation_vectorizer.transform([variation])
    text_vector = text_vectorizer.transform([text])
    input_vector = scipy.sparse.hstack((gene_vector, variation_vector, text_vector))

    # Prediction
    prediction = model.predict(input_vector)[0]

    # Get treatment info
    treatment_info = treatment_map.get(prediction, {})
    treatment_text = treatment_info.get("text", "No recommendation available")
    treatment_url = treatment_info.get("url", "#")

    return jsonify({
        "prediction": int(prediction),
        "treatment": treatment_text,
        "url": treatment_url
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint for monitoring"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Use environment variable for port with a default value
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
