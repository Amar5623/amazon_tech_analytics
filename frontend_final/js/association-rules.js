// API utility functions
const ML_BASE = 'http://localhost:8000/ml';

async function apiGet(endpoint, params = {}) {
    try {
        const queryString = Object.keys(params)
            .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
            .join('&');
        
        const url = `${endpoint}${queryString ? '?' + queryString : ''}`;
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        throw error;
    }
}

async function apiPost(endpoint, data = {}) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`Error posting to ${endpoint}:`, error);
        throw error;
    }
}

// ML API functions
const mlAPI = {
    async getDataStats() {
        return await apiGet(`${ML_BASE}/data-stats`);
    },
    
    async runAssociationRules(data) {
        return await apiPost(`${ML_BASE}/association-rules`, data);
    }
};

// Helper functions for ML visualizations
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

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    
    // Check if elements exist before adding event listeners
    const minSupportEl = document.getElementById('min-support');
    const minConfidenceEl = document.getElementById('min-confidence');
    const minLiftEl = document.getElementById('min-lift');
    const runAssociationEl = document.getElementById('run-association');
    
    if (minSupportEl) {
        minSupportEl.addEventListener('input', updateMinSupportValue);
        console.log("Added event listener to min-support");
    } else {
        console.error("Element with ID 'min-support' not found");
    }
    
    if (minConfidenceEl) {
        minConfidenceEl.addEventListener('input', updateMinConfidenceValue);
        console.log("Added event listener to min-confidence");
    } else {
        console.error("Element with ID 'min-confidence' not found");
    }
    
    if (minLiftEl) {
        minLiftEl.addEventListener('input', updateMinLiftValue);
        console.log("Added event listener to min-lift");
    } else {
        console.error("Element with ID 'min-lift' not found");
    }
    
    if (runAssociationEl) {
        runAssociationEl.addEventListener('click', runAssociationRules);
        console.log("Added event listener to run-association");
    } else {
        console.error("Element with ID 'run-association' not found");
    }
    
    // Initialize the page
    initializePage();
});

async function initializePage() {
    console.log("Initializing page...");
    try {
        // Set initial values for sliders
        updateMinSupportValue();
        updateMinConfidenceValue();
        updateMinLiftValue();
        
        // Load dataset statistics
        console.log("Fetching data stats...");
        const dataStats = await mlAPI.getDataStats();
        console.log("Data stats received:", dataStats);
        
        if (dataStats.success) {
            // Populate category dropdown
            if (dataStats.statistics && dataStats.statistics.categories) {
                populateCategoryDropdown(dataStats.statistics.categories);
            }
            
            // Enable run button
            const runButton = document.getElementById('run-association');
            if (runButton) {
                runButton.disabled = false;
            }
        }
    } catch (error) {
        console.error("Error initializing page:", error);
        showError('results-container', `Error initializing page: ${error.message}`);
    }
}

function populateCategoryDropdown(categories) {
    const select = document.getElementById('category-select');
    if (!select) {
        console.error("Category select element not found");
        return;
    }
    
    select.innerHTML = '<option value="">All Categories</option>';
    
    // Sort categories alphabetically
    const sortedCategories = [...categories].sort();
    
    sortedCategories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        select.appendChild(option);
    });
    
    console.log("Categories populated:", sortedCategories.length);
}

function updateMinSupportValue() {
    const supportEl = document.getElementById('min-support');
    const valueEl = document.getElementById('min-support-value');
    
    if (supportEl && valueEl) {
        const value = supportEl.value;
        valueEl.textContent = `${value} (${(value * 100).toFixed(0)}%)`;
    }
}

function updateMinConfidenceValue() {
    const confidenceEl = document.getElementById('min-confidence');
    const valueEl = document.getElementById('min-confidence-value');
    
    if (confidenceEl && valueEl) {
        const value = confidenceEl.value;
        valueEl.textContent = `${value} (${(value * 100).toFixed(0)}%)`;
    }
}

function updateMinLiftValue() {
    const liftEl = document.getElementById('min-lift');
    const valueEl = document.getElementById('min-lift-value');
    
    if (liftEl && valueEl) {
        const value = liftEl.value;
        valueEl.textContent = value;
    }
}

async function runAssociationRules() {
    console.log("Running association rules...");
    
    const runButton = document.getElementById('run-association');
    const resultsContainer = document.getElementById('results-container');
    
    if (!runButton || !resultsContainer) {
        console.error("Required elements not found");
        return;
    }
    
    // Show loading state
    runButton.disabled = true;
    runButton.textContent = 'Running...';
    resultsContainer.style.display = 'block';
    showLoading('results-container', 'Mining association rules...');
    
    try {
        // Get selected category
        const categorySelect = document.getElementById('category-select');
        const category = categorySelect ? categorySelect.value : '';
        
        // Get parameters
        const minSupportEl = document.getElementById('min-support');
        const minConfidenceEl = document.getElementById('min-confidence');
        const minLiftEl = document.getElementById('min-lift');
        const metricEl = document.getElementById('metric');
        
        const minSupport = minSupportEl ? parseFloat(minSupportEl.value) : 0.1;
        const minConfidence = minConfidenceEl ? parseFloat(minConfidenceEl.value) : 0.5;
        const minLift = minLiftEl ? parseFloat(minLiftEl.value) : 1.0;
        const metric = metricEl ? metricEl.value : 'lift';
        
        // Prepare request data
        const requestData = {
            min_support: minSupport,
            min_confidence: minConfidence,
            min_lift: minLift,
            metric: metric,
            category: category || undefined // Only include if not empty
        };
        
        console.log("Request data:", requestData);
        
        // Call API
        console.log("Calling API...");
        const response = await mlAPI.runAssociationRules(requestData);
        console.log("API response:", response);
        
        if (response.success) {
            displayAssociationRules(response);
        } else {
            showError('results-container', 'Failed to mine association rules');
        }
    } catch (error) {
        console.error("Error running association rules:", error);
        showError('results-container', `Error mining association rules: ${error.message}`);
    } finally {
        // Reset button state
        if (runButton) {
            runButton.disabled = false;
            runButton.textContent = 'Find Association Rules';
        }
    }
}

function displayAssociationRules(results) {
    console.log("Displaying association rules:", results);
    
    const container = document.getElementById('results-container');
    if (!container) {
        console.error("Results container not found");
        return;
    }
    
    container.innerHTML = '';
    
    // Display summary
    const summaryContainer = document.createElement('div');
    summaryContainer.className = 'rules-summary';
    summaryContainer.innerHTML = `
        <h4>Summary</h4>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">Total Rules Found</div>
                <div class="summary-value">${results.num_rules || 0}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Products Analyzed</div>
                <div class="summary-value">${results.num_products || 'N/A'}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Average Confidence</div>
                <div class="summary-value">${results.avg_confidence ? results.avg_confidence.toFixed(4) : 'N/A'}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Average Lift</div>
                <div class="summary-value">${results.avg_lift ? results.avg_lift.toFixed(4) : 'N/A'}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Average Support</div>
                <div class="summary-value">${results.avg_support ? results.avg_support.toFixed(4) : 'N/A'}</div>
            </div>
        </div>
    `;
    container.appendChild(summaryContainer);
    
    // Display visualizations
    if (results.visualizations) {
        const visContainer = document.createElement('div');
        visContainer.className = 'visualizations-container';
        visContainer.innerHTML = '<h4>Visualizations</h4><div id="visualizations"></div>';
        container.appendChild(visContainer);
        
        // Display each visualization
        Object.entries(results.visualizations).forEach(([name, base64String]) => {
            displayImage('visualizations', base64String, name);
        });
    }
    
    // Display rules table
    if (results.rules && results.rules.length > 0) {
        const rulesContainer = document.createElement('div');
        rulesContainer.className = 'rules-table-container';
        rulesContainer.innerHTML = `
            <h4>Association Rules</h4>
            <div class="table-controls">
                <input type="text" id="rules-search" placeholder="Search rules...">
                <select id="rules-sort">
                    <option value="lift-desc">Sort by Lift (High to Low)</option>
                    <option value="confidence-desc">Sort by Confidence (High to Low)</option>
                    <option value="support-desc">Sort by Support (High to Low)</option>
                </select>
            </div>
            <div id="rules-table"></div>
        `;
        container.appendChild(rulesContainer);
        
        // Create table
        const table = document.createElement('table');
        table.className = 'data-table';
        
        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        results.rules.forEach(rule => {
            const row = document.createElement('tr');
            
            // Antecedents
            const antecedentsCell = document.createElement('td');
            antecedentsCell.textContent = Array.isArray(rule.antecedents) ? rule.antecedents.join(', ') : rule.antecedents;
            row.appendChild(antecedentsCell);
            
            // Consequents
            const consequentsCell = document.createElement('td');
            consequentsCell.textContent = Array.isArray(rule.consequents) ? rule.consequents.join(', ') : rule.consequents;
            row.appendChild(consequentsCell);
            
            // Support
            const supportCell = document.createElement('td');
            supportCell.textContent = rule.support.toFixed(4);
            row.appendChild(supportCell);
            
            // Confidence
            const confidenceCell = document.createElement('td');
            confidenceCell.textContent = rule.confidence.toFixed(4);
            row.appendChild(confidenceCell);
            
            // Lift
            const liftCell = document.createElement('td');
            liftCell.textContent = rule.lift.toFixed(4);
            row.appendChild(liftCell);
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        
        const rulesTableDiv = document.getElementById('rules-table');
        if (rulesTableDiv) {
            rulesTableDiv.appendChild(table);
            
            // Add event listeners for search and sort
            const searchInput = document.getElementById('rules-search');
            const sortSelect = document.getElementById('rules-sort');
            
            if (searchInput) {
                searchInput.addEventListener('input', filterRules);
            }
            
            if (sortSelect) {
                sortSelect.addEventListener('change', sortRules);
            }
        }
    }    else {
        const noRulesDiv = document.createElement('div');
        noRulesDiv.className = 'no-rules';
        noRulesDiv.textContent = 'No association rules found with the current parameters. Try lowering the minimum support or confidence.';
        container.appendChild(noRulesDiv);
    }


function filterRules() {
    const searchTerm = document.getElementById('rules-search').value.toLowerCase();
    const rows = document.querySelectorAll('#rules-table tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

function sortRules() {
    const sortValue = document.getElementById('rules-sort').value;
    const tbody = document.querySelector('#rules-table tbody');
    
    if (!tbody) {
        console.error("Table body not found");
        return;
    }
    
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // Determine column index and sort direction
    let columnIndex;
    let sortDirection = 1; // 1 for descending, -1 for ascending
    
    if (sortValue === 'lift-desc') {
        columnIndex = 4;
    } else if (sortValue === 'confidence-desc') {
        columnIndex = 3;
    } else if (sortValue === 'support-desc') {
        columnIndex = 2;
    }
    
    // Sort rows
    rows.sort((a, b) => {
        const aValue = parseFloat(a.cells[columnIndex].textContent);
        const bValue = parseFloat(b.cells[columnIndex].textContent);
        return sortDirection * (bValue - aValue);
    });
    
    // Reappend rows in new order
    rows.forEach(row => tbody.appendChild(row));
}

}