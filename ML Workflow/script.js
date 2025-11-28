const steps = [
    {
        title: "Data Loading",
        description: "This is the foundational step where raw data is ingested into the system. We load the dataset from a CSV file, handling initial parsing and type inference. This ensures the data is available in a structured format (DataFrame) for downstream processing.",
        type: "data",
        dataPreview: [
            ["Customer ID", "Credit Limit", "Utilisation %", "Avg Payment Ratio", "Min Due Paid Frequency"],
            ["C001", "165000", "12", "32", "66"],
            ["C002", "95000", "10", "49", "45"],
            ["C003", "60000", "14", "88", "23"],
            ["C004", "125000", "99", "65", "31"],
            ["C005", "115000", "23", "48", "46"]
        ],
        downloadLink: "Sample.csv"
    },
    {
        title: "Feature Engineering",
        description: "Here we transform raw data into meaningful predictors. We create 14+ early warning signals, which are derived features designed to capture subtle patterns of risk (e.g., utilization trends, payment gaps) that raw variables might miss.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Data Preparation",
        description: "This step cleans and structures the data for modeling. We select the relevant features created in the previous step and define the binary target variable (e.g., 1 for delinquency, 0 for non-delinquency), ensuring the dataset is strictly numerical and ready for algorithms.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Train-Test Split",
        description: "To evaluate the model fairly, we split the data into a training set (80%) and a testing set (20%). We use stratification to ensure that the proportion of delinquent cases is preserved in both sets, preventing bias in evaluation.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Feature Scaling",
        description: "Machine learning models often perform better when features are on the same scale. We use StandardScaler to normalize the features (mean=0, variance=1), preventing features with larger magnitudes from dominating the model learning process.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Model Training",
        description: "This is the core learning phase. We instantiate a Random Forest Classifier with 100 decision trees. The ensemble method learns complex non-linear relationships in the training data to distinguish between safe and risky customers.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Model Evaluation",
        description: "We assess how well the model performs on unseen data (the test set). Metrics like Accuracy, Precision, Recall, and the Confusion Matrix provide a comprehensive view of the model's reliability and its ability to minimize false positives and false negatives.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Feature Importance",
        description: "Understanding 'why' a model makes a decision is crucial. We extract feature importance scores to identify which variables (e.g., 'Recent Delinquency', 'Utilization Ratio') are the most predictive of risk, offering business insights.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Making Predictions",
        description: "The operational goal of the workflow. We use the trained and calibrated model to score new, unseen customer profiles, predicting their likelihood of delinquency to inform credit decisions.",
        image: "ml_workflow_abstract_1764339804100.png"
    },
    {
        title: "Complete Code Flow",
        description: "This represents the end-to-end pipeline execution. It orchestrates all previous steps in a sequence, ensuring reproducibility, automation, and a streamlined path from raw data to actionable insights.",
        image: "ml_workflow_abstract_1764339804100.png"
    }
];

const flowchartContainer = document.getElementById('flowchart');
const stepTitle = document.getElementById('step-title');
const stepDescription = document.getElementById('step-description');
const rightPanel = document.getElementById('right-panel');

// Render Flowchart
steps.forEach((step, index) => {
    const nodeWrapper = document.createElement('div');
    nodeWrapper.className = 'node-wrapper';
    nodeWrapper.style.animationDelay = `${index * 0.1}s`;

    const node = document.createElement('div');
    node.className = 'node';
    node.innerHTML = `<h3>${step.title}</h3>`;
    node.addEventListener('click', () => activateStep(step, node));

    nodeWrapper.appendChild(node);

    // Add connector if not the last step
    if (index < steps.length - 1) {
        const connector = document.createElement('div');
        connector.className = 'connector';
        nodeWrapper.appendChild(connector);
    }

    flowchartContainer.appendChild(nodeWrapper);
});

function activateStep(step, activeNode) {
    // Update Active State in Flowchart
    document.querySelectorAll('.node').forEach(n => n.classList.remove('active'));
    activeNode.classList.add('active');

    // Update Left Panel (Description)
    stepTitle.style.opacity = 0;
    stepDescription.style.opacity = 0;

    setTimeout(() => {
        stepTitle.textContent = step.title;
        stepDescription.textContent = step.description;
        stepTitle.style.opacity = 1;
        stepDescription.style.opacity = 1;
    }, 200);

    // Update Right Panel
    // Clear current content
    rightPanel.innerHTML = '';

    if (step.type === 'data') {
        renderDataPreview(step);
    } else if (step.image) {
        renderImage(step.image);
    } else {
        renderPlaceholder();
    }
}

function renderDataPreview(step) {
    const container = document.createElement('div');
    container.className = 'data-preview-container';

    // Create Table
    const tableWrapper = document.createElement('div');
    tableWrapper.className = 'table-wrapper';
    const table = document.createElement('table');

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    step.dataPreview[0].forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    step.dataPreview.slice(1).forEach(rowData => {
        const tr = document.createElement('tr');
        rowData.forEach(cellData => {
            const td = document.createElement('td');
            td.textContent = cellData;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableWrapper.appendChild(table);
    container.appendChild(tableWrapper);

    // Download Button
    if (step.downloadLink) {
        const btn = document.createElement('a');
        btn.href = step.downloadLink;
        btn.className = 'download-btn';
        btn.innerHTML = '<span>📥</span> Download Full Dataset';
        btn.setAttribute('download', '');
        container.appendChild(btn);
    }

    rightPanel.appendChild(container);

    // Fade in
    container.style.opacity = 0;
    setTimeout(() => container.style.opacity = 1, 100);
}

function renderImage(imageSrc) {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-wrapper';

    const img = document.createElement('img');
    img.src = imageSrc;
    img.alt = "Step Visualization";

    wrapper.appendChild(img);
    rightPanel.appendChild(wrapper);

    // Fade in
    img.style.opacity = 0;
    setTimeout(() => img.style.opacity = 1, 100);
}

function renderPlaceholder() {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-wrapper';

    const placeholder = document.createElement('div');
    placeholder.className = 'placeholder';
    placeholder.innerHTML = `
        <div class="icon">📊</div>
        <p>Visualization</p>
    `;

    wrapper.appendChild(placeholder);
    rightPanel.appendChild(wrapper);
}

// Activate first step by default after a delay
setTimeout(() => {
    const firstNode = document.querySelector('.node');
    if (firstNode) {
        activateStep(steps[0], firstNode);
    }
}, 1000);
