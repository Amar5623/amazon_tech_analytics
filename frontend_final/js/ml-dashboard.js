import { api, mlAPI, showLoading, showError, displayImage } from './api.js';

document.addEventListener('DOMContentLoaded', function() {
    // Load dataset statistics
    loadDataStats();
});

async function loadDataStats() {
    const container = document.getElementById('data-stats-container');
    
    try {
        showLoading('data-stats-container', 'Loading dataset statistics...');
        
        const response = await mlAPI.getDataStats();
        
        if (response.success) {
            // Create a dashboard with the statistics
            const stats = response.statistics;
            const specs = response.specifications;
            
            let html = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>Products</h4>
                        <div class="stat-value">${stats.num_products.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Categories</h4>
                        <div class="stat-value">${stats.num_categories.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Brands</h4>
                        <div class="stat-value">${stats.num_brands.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Price Range</h4>
                        <div class="stat-value">$${stats.price_range[0].toFixed(2)} - $${stats.price_range[1].toFixed(2)}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Rating Range</h4>
                        <div class="stat-value">${stats.rating_range[0].toFixed(1)} - ${stats.rating_range[1].toFixed(1)}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Review Count Range</h4>
                        <div class="stat-value">${stats.review_count_range[0]} - ${stats.review_count_range[1].toLocaleString()}</div>
                    </div>
                </div>
                
                <div class="specs-section">
                    <h4>Top Product Specifications</h4>
                    <div class="specs-list">
                        ${specs.top_specs.map(spec => `
                            <div class="spec-item">
                                <span class="spec-name">${spec.name}</span>
                                <span class="spec-count">${spec.count.toLocaleString()} products</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            
            container.innerHTML = html;
            
            // If there are visualizations, display them
            if (response.visualizations) {
                const visContainer = document.createElement('div');
                visContainer.className = 'visualizations-section';
                visContainer.innerHTML = '<h4>Data Visualizations</h4><div id="data-visualizations"></div>';
                container.appendChild(visContainer);
                
                // Display each visualization
                Object.entries(response.visualizations).forEach(([name, base64String]) => {
                    displayImage('data-visualizations', base64String, name);
                });
            }
        } else {
            showError('data-stats-container', 'Failed to load dataset statistics');
        }
    } catch (error) {
        console.error('Error loading data stats:', error);
        showError('data-stats-container', `Error loading dataset statistics: ${error.message}`);
    }
}
