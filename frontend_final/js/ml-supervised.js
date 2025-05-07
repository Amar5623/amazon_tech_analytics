// frontend_final\js\ml-supervised.js




document.addEventListener('DOMContentLoaded', function() {
    // Initialize the page
    initializePage();
    
    // Set up event listeners
    document.getElementById('target-column').addEventListener('change', onTargetColumnChange);
    document.getElementById('algorithm-type').addEventListener('change', onAlgorithmTypeChange);
    document.getElementById('algorithm').addEventListener('change', onAlgorithmChange);
    document.getElementById('test-size').addEventListener('input', updateTestSizeValue);
    document.getElementById('run-algorithm').addEventListener('click', runAlgorithm);
});

async function initializePage() {
    try {
        // Load available algorithms
        const response = await mlAPI.getAvailableAlgorithms();
        
        if (response.success) {
            // Store algorithms for later use
            window.availableAlgorithms = response.algorithms;
        }
    } catch (error) {
        showError('algorithm-params', `Error initializing page: ${error.message}`);
    }
}

function onTargetColumnChange() {
    const targetColumn = document.getElementById('target-column').value;
    
    // Reset algorithm type and algorithm
    document.getElementById('algorithm-type').value = '';
    document.getElementById('algorithm').value = '';
    document.getElementById('algorithm').disabled = true;
    document.getElementById('algorithm-params').innerHTML = '';
    document.getElementById('run-algorithm').disabled = true;
    
    if (!targetColumn) return;
    
    // Enable algorithm type selection
    document.getElementById('algorithm-type').disabled = false;
}

function onAlgorithmTypeChange() {
    const algorithmType = document.getElementById('algorithm-type').value;
    const targetColumn = document.getElementById('target-column').value;
    
    // Reset algorithm and params
    document.getElementById('algorithm').value = '';
    document.getElementById('algorithm-params').innerHTML = '';
    document.getElementById('run-algorithm').disabled = true;
    
    if (!algorithmType || !targetColumn) {
        document.getElementById('algorithm').disabled = true;
        return;
    }
    
    // Populate algorithm dropdown based on type
    populateAlgorithmDropdown(algorithmType);
    
    // Enable algorithm selection
    document.getElementById('algorithm').disabled = false;
}

function populateAlgorithmDropdown(algorithmType) {
    const select = document.getElementById('algorithm');
    
    // Clear existing options except the first one
    while (select.options.length > 1) {
        select.remove(1);
    }
    
    if (!window.availableAlgorithms) return;
    
    // Get algorithms based on type
    const algorithms = algorithmType === 'regression' 
        ? window.availableAlgorithms.supervised.regression
        : window.availableAlgorithms.supervised.classification;
    
    // Add new options
    algorithms.forEach(algo => {
        const option = document.createElement('option');
        option.value = algo.name;
        option.textContent = algo.display_name;
        select.appendChild(option);
    });
}

async function onAlgorithmChange() {
    const algorithm = document.getElementById('algorithm').value;
    const algorithmType = document.getElementById('algorithm-type').value;
    const targetColumn = document.getElementById('target-column').value;
    
    if (!algorithm || !algorithmType || !targetColumn) {
        document.getElementById('algorithm-params').innerHTML = '';
        document.getElementById('run-algorithm').disabled = true;
        return;
    }
    
    try {
        // Check if algorithm is compatible with the target column
        const compatibility = await mlAPI.checkModelCompatibility(algorithm, targetColumn);
        
        if (!compatibility.compatible) {
            showError('algorithm-params', `This algorithm is not compatible with the selected target column: ${compatibility.reasons.join(', ')}`);
            document.getElementById('run-algorithm').disabled = true;
            return;
        }
        
        // Get algorithm details
        const algorithms = algorithmType === 'regression' 
            ? window.availableAlgorithms.supervised.regression
            : window.availableAlgorithms.supervised.classification;
        
        const selectedAlgo = algorithms.find(algo => algo.name === algorithm);
        
        if (selectedAlgo) {
            // Populate algorithm parameters
            populateAlgorithmParams(selectedAlgo.parameters);
            
            // Enable run button
            document.getElementById('run-algorithm').disabled = false;
        }
    } catch (error) {
        showError('algorithm-params', `Error checking algorithm compatibility: ${error.message}`);
        document.getElementById('run-algorithm').disabled = true;
    }
}

function populateAlgorithmParams(parameters) {
    const container = document.getElementById('algorithm-params');
    container.innerHTML = '<h4>Algorithm Parameters</h4>';
    
    if (!parameters || parameters.length === 0) {
        container.innerHTML += '<p>No parameters available for this algorithm</p>';
        return;
    }
    
    parameters.forEach(param => {
        const div = document.createElement('div');
        div.className = 'param-item';
        
        const label = document.createElement('label');
        label.htmlFor = `param-${param.name}`;
        label.textContent = param.name.replace('_', ' ');
        
        let input;
        
        if (param.type === 'float' || param.type === 'int') {
            input = document.createElement('input');
            input.type = 'number';
            input.id = `param-${param.name}`;
            input.name = `params.${param.name}`;
            input.value = param.default;
            input.step = param.type === 'float' ? '0.01' : '1';
        } else if (param.type === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `param-${param.name}`;
            input.name = `params.${param.name}`;
            input.checked = param.default;
        } else if (Array.isArray(param.type)) {
            input = document.createElement('select');
            input.id = `param-${param.name}`;
            input.name = `params.${param.name}`;
            
            param.type.forEach(option => {
                const optionEl = document.createElement('option');
                optionEl.value = option;
                optionEl.textContent = option;
                optionEl.selected = option === param.default;
                input.appendChild(optionEl);
            });
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.id = `param-${param.name}`;
            input.name = `params.${param.name}`;
            input.value = param.default;
        }
        
        div.appendChild(label);
        div.appendChild(input);
        
        if (param.description) {
            const description = document.createElement('small');
            description.className = 'param-description';
            description.textContent = param.description;
            div.appendChild(description);
        }
        
        container.appendChild(div);
    });
}

function updateTestSizeValue() {
    const value = document.getElementById('test-size').value;
    document.getElementById('test-size-value').textContent = `${value} (${(value * 100).toFixed(0)}%)`;
}

async function runAlgorithm() {
    // Show loading state
    document.getElementById('run-algorithm').disabled = true;
    document.getElementById('run-algorithm').textContent = 'Running...';
    showLoading('results-container', 'Running algorithm...');
    document.getElementById('results-container').style.display = 'block';
    
    try {
        // Get form values
        const targetColumn = document.getElementById('target-column').value;
        const algorithm = document.getElementById('algorithm').value;
        const testSize = parseFloat(document.getElementById('test-size').value);
        
        // Get algorithm parameters
        const params = {};
        document.querySelectorAll('[name^="params."]').forEach(input => {
            const paramName = input.name.replace('params.', '');
            let value;
            
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                value = input.value === '' ? null : Number(input.value);
            } else {
                value = input.value;
            }
            
            params[paramName] = value;
        });
        
        // Prepare request data
        const requestData = {
            target_column: targetColumn,
            algorithm: algorithm,
            params: params,
            test_size: testSize,
            features: [] // The backend will use all available features
        };
        
        // Call API
        const response = await mlAPI.runSupervisedLearning(requestData);
        
        if (response.success) {
            displayResults(response);
        } else {
            showError('results-container', 'Failed to run algorithm');
        }
    } catch (error) {
        showError('results-container', `Error running algorithm: ${error.message}`);
    } finally {
        // Reset button state
        document.getElementById('run-algorithm').disabled = false;
        document.getElementById('run-algorithm').textContent = 'Run Algorithm';
    }
}

function displayResults(results) {
    const container = document.getElementById('results-container');
    container.innerHTML = '<h3>Results</h3>';
    
    // Display metrics
    const metricsContainer = document.createElement('div');
    metricsContainer.className = 'metrics-container';
    metricsContainer.innerHTML = '<h4>Performance Metrics</h4>';
    
    const metricsDisplay = document.createElement('div');
    metricsDisplay.className = 'metrics-grid';
    
    if (results.metrics) {
        Object.entries(results.metrics).forEach(([name, value]) => {
            const metricDiv = document.createElement('div');
            metricDiv.className = 'metric-item';
            metricDiv.innerHTML = `
                <div class="metric-name">${name.replace('_', ' ')}</div>
                <div class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</div>
            `;
            metricsDisplay.appendChild(metricDiv);
        });
    } else {
        metricsDisplay.innerHTML = '<p>No metrics available</p>';
    }
    
    metricsContainer.appendChild(metricsDisplay);
    container.appendChild(metricsContainer);
    
    // Display visualizations
    if (results.visualizations) {
        const visContainer = document.createElement('div');
        visContainer.className = 'visualizations-container';
        visContainer.innerHTML = '<h4>Visualizations</h4><div id="result-visualizations"></div>';
        container.appendChild(visContainer);
        
        // Display each visualization
        Object.entries(results.visualizations).forEach(([name, base64String]) => {
            displayImage('result-visualizations', base64String, name);
        });
    }
    
    // Display feature importance
    if (results.feature_importance) {
        const fiContainer = document.createElement('div');
        fiContainer.className = 'feature-importance-container';
        fiContainer.innerHTML = '<h4>Feature Importance</h4>';
        
        const fiTable = document.createElement('table');
        fiTable.className = 'feature-importance-table';
        
        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        const featureHeader = document.createElement('th');
        featureHeader.textContent = 'Feature';
        headerRow.appendChild(featureHeader);
        
        const importanceHeader = document.createElement('th');
        importanceHeader.textContent = 'Importance';
        headerRow.appendChild(importanceHeader);
        
        thead.appendChild(headerRow);
        fiTable.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        // Sort features by importance
        const sortedFeatures = Object.entries(results.feature_importance)
            .sort((a, b) => b[1] - a[1]);
        
        sortedFeatures.forEach(([feature, importance]) => {
            const row = document.createElement('tr');
            
            const featureCell = document.createElement('td');
            featureCell.textContent = feature;
            row.appendChild(featureCell);
            
            const importanceCell = document.createElement('td');
            
            // Create a visual bar
            const barContainer = document.createElement('div');
            barContainer.className = 'importance-bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'importance-bar';
            bar.style.width = `${importance * 100}%`;
            
            const value = document.createElement('span');
            value.className = 'importance-value';
            value.textContent = importance.toFixed(4);
            
            barContainer.appendChild(bar);
            barContainer.appendChild(value);
            
            importanceCell.appendChild(barContainer);
            row.appendChild(importanceCell);
            
            tbody.appendChild(row);
        });
        
        fiTable.appendChild(tbody);
        fiContainer.appendChild(fiTable);
        container.appendChild(fiContainer);
    }
    
    // Display detailed report if available
    if (results.detailed_report) {
        const reportContainer = document.createElement('div');
        reportContainer.className = 'detailed-report-container';
        reportContainer.innerHTML = '<h4>Detailed Report</h4>';
        
        const reportPre = document.createElement('pre');
        reportPre.className = 'detailed-report';
        
        if (typeof results.detailed_report === 'string') {
            reportPre.textContent = results.detailed_report;
        } else {
            reportPre.textContent = JSON.stringify(results.detailed_report, null, 2);
        }
        
        reportContainer.appendChild(reportPre);
        container.appendChild(reportContainer);
    }
}

// Import helper functions from api.js
function showLoading(elementId, message = 'Loading...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="loading">${message}</div>`;
    }
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="error-message">${message}</div>`;
    }
}

function displayImage(containerId, base64String, altText = 'Visualization') {
    const container = document.getElementById(containerId);
    if (container && base64String) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${base64String}`;
        img.alt = altText;
        img.className = 'visualization-image';
        container.appendChild(img);
    }
}

