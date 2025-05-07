// Import the necessary functions and objects from the API module
import { mlAPI, showLoading, showError, displayImage2 } from './api.js';

document.addEventListener('DOMContentLoaded', async function() {
    // Initialize the page
    await initFeatureImportancePage();
    
    // Add event listener for the run button
    document.getElementById('run-feature-importance').addEventListener('click', runFeatureImportanceAnalysis);
});

/** 
 * Initialize the feature importance page 
 */
async function initFeatureImportancePage() {
    try {
        // Use the valid columns directly instead of fetching them
        // These are the columns available in the backend dataset
        const validColumns = [
            'price',
            'rating',
            'review_count',
            'category',
        ];
        
        // Populate the dropdown with these valid columns
        populateTargetColumnOptions(validColumns);
    } catch (error) {
        console.error('Error initializing feature importance page:', error);
        showError('feature-importance-container', 'Failed to initialize page: ' + error.message);
    }
}

/** 
 * Populate the target column dropdown with available columns 
 */
function populateTargetColumnOptions(columns) {
    if (!columns || !Array.isArray(columns)) {
        console.error("Invalid columns data:", columns);
        showError('feature-importance-container', 'Invalid column data received');
        return;
    }
    
    const targetColumnSelect = document.getElementById('target-column');
    
    // Clear existing options
    targetColumnSelect.innerHTML = '';
    
    // Add a default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select Target Column --';
    targetColumnSelect.appendChild(defaultOption);
    
    // Add options for each column
    columns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        targetColumnSelect.appendChild(option);
    });
    
    // If no options were added (besides the default), show an error
    if (targetColumnSelect.options.length <= 1) {
        showError('feature-importance-container', 'No valid columns found in data');
    }
}

/** 
 * Run the feature importance analysis 
 */
async function runFeatureImportanceAnalysis() {
    const targetColumn = document.getElementById('target-column').value;
    
    if (!targetColumn) {
        showError('feature-importance-container', 'Please select a target column');
        return;
    }
    
    try {
        // Show loading indicator
        showLoading('feature-importance-container', 'Running feature importance analysis...');
        
        // Clear previous results
        document.getElementById('feature-table-container').innerHTML = '';
        
        // Call the API to get feature importance
        const result = await mlAPI.getFeatureImportance(targetColumn);
        
        // Log the result to see its structure
        console.log("Feature importance result:", result);
        
        if (result.success) {
            // Display the results
            displayFeatureImportanceResults(result);
        } else {
            showError('feature-importance-container', 'Failed to run feature importance analysis');
        }
    } catch (error) {
        console.error('Error running feature importance analysis:', error);
        showError('feature-importance-container', 'Error: ' + error.message);
    }
}

/** 
 * Display the feature importance results 
 */
function displayFeatureImportanceResults(result) {
    const container = document.getElementById('feature-importance-container');
    const tableContainer = document.getElementById('feature-table-container');
    
    // Clear loading indicator and previous results
    container.innerHTML = '';
    tableContainer.innerHTML = '';
    
    // Display model information
    const infoDiv = document.createElement('div');
    infoDiv.className = 'model-info';
    infoDiv.innerHTML = `
        <p><strong>Target Column:</strong> ${result.target_column}</p>
        <p><strong>Model Type:</strong> ${result.model_type || 'Random Forest'}</p>
    `;
    container.appendChild(infoDiv);
    
    // Display API-provided visualization if available
    if (result.visualization) {
        const visualTitle = document.createElement('h3');
        visualTitle.textContent = 'Feature Importance Visualization';
        container.appendChild(visualTitle);
        
        // Use the displayImage function to show the base64 image directly in the container
        displayImage2('feature-importance-container', result.visualization, 'Feature Importance Visualization');
    }
    
    // Check if feature_importance exists and is an array
    if (!result.feature_importance || !Array.isArray(result.feature_importance)) {
        showError('feature-table-container', 'No feature importance data available');
        return;
    }
    
    // Create and display the feature importance table
    const tableTitle = document.createElement('h3');
    tableTitle.textContent = 'Feature Importance Details';
    tableContainer.appendChild(tableTitle);
    
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Feature</th>
            <th>Importance</th>
            <th>Visualization</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Sort features by importance (descending)
    const sortedFeatures = [...result.feature_importance].sort((a, b) => b.importance - a.importance);
    
    // Add rows for each feature
    sortedFeatures.forEach(feature => {
        const tr = document.createElement('tr');
        
        // Calculate percentage for the bar width
        const percentage = (feature.importance * 100).toFixed(2);
        
        tr.innerHTML = `
            <td>${feature.feature}</td>
            <td>${feature.importance.toFixed(4)}</td>
            <td>
                <div class="importance-bar-container">
                    <div class="importance-bar" style="width: ${percentage}%"></div>
                    <span class="importance-value">${percentage}%</span>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
    
    table.appendChild(tbody);
    tableContainer.appendChild(table);
}
