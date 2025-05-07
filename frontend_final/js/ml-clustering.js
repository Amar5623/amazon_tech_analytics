document.addEventListener('DOMContentLoaded', function() {
    // Initialize the page
    initializePage();
    
    // Set up event listeners
    document.getElementById('clustering-algorithm').addEventListener('change', onAlgorithmChange);
    document.getElementById('n-clusters').addEventListener('input', updateNClustersValue);
    document.getElementById('run-clustering').addEventListener('click', runClustering);
});

async function initializePage() {
    try {
        // Load available algorithms
        const response = await mlAPI.getAvailableAlgorithms();
        
        if (response.success) {
            populateAlgorithmDropdown(response.algorithms.unsupervised.clustering);
        }
        
        // Load available features
        const dataStats = await mlAPI.getDataStats();
        if (dataStats.success) {
            populateFeatureCheckboxes(dataStats);
        }
    } catch (error) {
        showError('algorithm-params', `Error initializing page: ${error.message}`);
    }
}

function populateAlgorithmDropdown(algorithms) {
    const select = document.getElementById('clustering-algorithm');
    
    // Clear existing options except the first one
    while (select.options.length > 1) {
        select.remove(1);
    }
    
    // Add new options
    algorithms.forEach(algo => {
        const option = document.createElement('option');
        option.value = algo.name;
        option.textContent = algo.display_name;
        select.appendChild(option);
    });
}

function populateFeatureCheckboxes(dataStats) {
    const container = document.getElementById('feature-checkboxes');
    container.innerHTML = '';
    
    // Add numeric features
    const numericFeatures = ['price', 'rating', 'review_count'];
    numericFeatures.forEach(feature => {
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = `feature-${feature}`;
        input.name = 'features';
        input.value = feature;
        input.checked = true;
        
        const label = document.createElement('label');
        label.htmlFor = `feature-${feature}`;
        label.textContent = feature.charAt(0).toUpperCase() + feature.slice(1).replace('_', ' ');
        
        div.appendChild(input);
        div.appendChild(label);
        container.appendChild(div);
    });
    
    // Add category and brand as features
    const categoricalFeatures = ['category', 'brand'];
    categoricalFeatures.forEach(feature => {
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = `feature-${feature}`;
        input.name = 'features';
        input.value = feature;
        input.checked = true;
        
        const label = document.createElement('label');
        label.htmlFor = `feature-${feature}`;
        label.textContent = feature.charAt(0).toUpperCase() + feature.slice(1);
        
        div.appendChild(input);
        div.appendChild(label);
        container.appendChild(div);
    });
    
    // Add top specifications as features
    if (dataStats.specifications && dataStats.specifications.top_specs) {
        const topSpecs = dataStats.specifications.top_specs.slice(0, 5); // Get top 5 specs
        
        if (topSpecs.length > 0) {
            const specHeader = document.createElement('div');
            specHeader.className = 'checkbox-header';
            specHeader.textContent = 'Top Specifications:';
            container.appendChild(specHeader);
            
            topSpecs.forEach(spec => {
                const div = document.createElement('div');
                div.className = 'checkbox-item';
                
                const input = document.createElement('input');
                input.type = 'checkbox';
                input.id = `feature-spec-${spec.name}`;
                input.name = 'features';
                input.value = `spec.${spec.name}`;
                
                const label = document.createElement('label');
                label.htmlFor = `feature-spec-${spec.name}`;
                label.textContent = spec.name;
                
                div.appendChild(input);
                div.appendChild(label);
                container.appendChild(div);
            });
        }
    }
    
    // Enable the run button if we have features
    document.getElementById('run-clustering').disabled = false;
}

async function onAlgorithmChange() {
    const algorithm = document.getElementById('clustering-algorithm').value;
    
    if (!algorithm) {
        document.getElementById('algorithm-params').innerHTML = '';
        document.getElementById('n-clusters-container').style.display = 'none';
        document.getElementById('run-clustering').disabled = true;
        return;
    }
    
    try {
        // Get algorithm details
        const response = await mlAPI.getAvailableAlgorithms();
        
        if (response.success) {
            const algorithms = response.algorithms.unsupervised.clustering;
            const selectedAlgo = algorithms.find(algo => algo.name === algorithm);
            
            if (selectedAlgo) {
                // Show/hide n_clusters based on algorithm
                if (algorithm === 'kmeans' || algorithm === 'hierarchical') {
                    document.getElementById('n-clusters-container').style.display = 'block';
                } else {
                    document.getElementById('n-clusters-container').style.display = 'none';
                }
                
                // Populate algorithm parameters
                populateAlgorithmParams(selectedAlgo.parameters);
                
                // Enable run button
                document.getElementById('run-clustering').disabled = false;
            }
        }
    } catch (error) {
        showError('algorithm-params', `Error loading algorithm parameters: ${error.message}`);
    }
}

function populateAlgorithmParams(parameters) {
    const container = document.getElementById('algorithm-params');
    container.innerHTML = '<h4>Algorithm Parameters</h4>';
    
    parameters.forEach(param => {
        // Skip n_clusters as we have a dedicated control for it
        if (param.name === 'n_clusters') return;
        
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

function updateNClustersValue() {
    const value = document.getElementById('n-clusters').value;
    document.getElementById('n-clusters-value').textContent = value;
}

async function runClustering() {
    // Show loading state
    document.getElementById('run-clustering').disabled = true;
    document.getElementById('run-clustering').textContent = 'Running...';
    showLoading('results-container', 'Running clustering algorithm...');
    document.getElementById('results-container').style.display = 'block';
    
    try {
        // Get selected algorithm
        const algorithm = document.getElementById('clustering-algorithm').value;
        
        // Get n_clusters if applicable
        let n_clusters = null;
        if (algorithm === 'kmeans' || algorithm === 'hierarchical') {
            n_clusters = parseInt(document.getElementById('n-clusters').value);
        }
        
        // Get selected features
        const featureCheckboxes = document.querySelectorAll('input[name="features"]:checked');
        const features = Array.from(featureCheckboxes).map(cb => cb.value);
        
        if (features.length === 0) {
            throw new Error('Please select at least one feature for clustering');
        }
        
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
            algorithm: algorithm,
            features: features,
            params: params
        };
        
        // Add n_clusters if applicable
        if (n_clusters !== null) {
            requestData.n_clusters = n_clusters;
        }
        
        // Call API
        const response = await mlAPI.runClustering(requestData);
        
        if (response.success) {
            displayClusteringResults(response);
        } else {
            showError('results-container', 'Failed to run clustering algorithm');
        }
    } catch (error) {
        showError('results-container', `Error running clustering: ${error.message}`);
    } finally {
        // Reset button state
        document.getElementById('run-clustering').disabled = false;
        document.getElementById('run-clustering').textContent = 'Run Clustering';
    }
}

function displayClusteringResults(results) {
    const container = document.getElementById('results-container');
    container.innerHTML = '<h3>Clustering Results</h3>';
    
    // Display metrics
    const metricsContainer = document.createElement('div');
    metricsContainer.className = 'metrics-container';
    metricsContainer.innerHTML = '<h4>Cluster Metrics</h4>';
    
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
        visContainer.innerHTML = '<h4>Cluster Visualizations</h4><div id="cluster-visualizations"></div>';
        container.appendChild(visContainer);
        
        // Display each visualization
        Object.entries(results.visualizations).forEach(([name, base64String]) => {
            displayImage('cluster-visualizations', base64String, name);
        });
    }
    
    // Display cluster statistics
    if (results.cluster_statistics) {
        const statsContainer = document.createElement('div');
        statsContainer.className = 'cluster-stats-container';
        statsContainer.innerHTML = '<h4>Cluster Statistics</h4>';
        
        const statsTable = document.createElement('table');
        statsTable.className = 'cluster-stats-table';
        
        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        // Add Feature column
        const featureHeader = document.createElement('th');
        featureHeader.textContent = 'Feature';
        headerRow.appendChild(featureHeader);
        
        // Add a column for each cluster
        const numClusters = results.num_clusters;
        for (let i = 0; i < numClusters; i++) {
            const clusterHeader = document.createElement('th');
            clusterHeader.textContent = `Cluster ${i}`;
            headerRow.appendChild(clusterHeader);
        }
        
        thead.appendChild(headerRow);
        statsTable.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        // Get all features from the first cluster
        const firstCluster = results.cluster_statistics['0'];
        const features = Object.keys(firstCluster);
        
        // Create a row for each feature
        features.forEach(feature => {
            const row = document.createElement('tr');
            
            // Add feature name
            const featureCell = document.createElement('td');
            featureCell.textContent = feature;
            row.appendChild(featureCell);
            
            // Add stats for each cluster
            for (let i = 0; i < numClusters; i++) {
                const clusterStats = results.cluster_statistics[i.toString()];
                const featureStats = clusterStats[feature];
                
                const cell = document.createElement('td');
                
                if (featureStats) {
                    // For numerical features, show mean and range
                    if (typeof featureStats.mean === 'number') {
                        cell.innerHTML = `
                            <div>Mean: ${featureStats.mean.toFixed(2)}</div>
                            <div>Min: ${featureStats.min.toFixed(2)}</div>
                            <div>Max: ${featureStats.max.toFixed(2)}</div>
                            <div>Count: ${featureStats.count}</div>
                        `;
                    } else {
                        // For categorical features, show count
                        cell.textContent = `Count: ${featureStats.count}`;
                    }
                } else {
                    cell.textContent = 'N/A';
                }
                
                row.appendChild(cell);
            }
            
            tbody.appendChild(row);
        });
        
        statsTable.appendChild(tbody);
        statsContainer.appendChild(statsTable);
        container.appendChild(statsContainer);
    }
    
    // Display cluster assignments summary
    if (results.cluster_assignments && results.product_ids) {
        const assignmentsContainer = document.createElement('div');
        assignmentsContainer.className = 'cluster-assignments-container';
        assignmentsContainer.innerHTML = '<h4>Cluster Assignments</h4>';
        
        // Create a summary of how many products in each cluster
        const clusterCounts = {};
        results.cluster_assignments.forEach(cluster => {
            clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
        });
        
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'cluster-summary';
        
        Object.entries(clusterCounts).sort((a, b) => a[0] - b[0]).forEach(([cluster, count]) => {
            const clusterDiv = document.createElement('div');
            clusterDiv.className = 'cluster-count';
            clusterDiv.innerHTML = `
                <div class="cluster-label">Cluster ${cluster}</div>
                <div class="cluster-value">${count} products</div>
            `;
            summaryDiv.appendChild(clusterDiv);
        });
        
        assignmentsContainer.appendChild(summaryDiv);
        container.appendChild(assignmentsContainer);
    }
}

