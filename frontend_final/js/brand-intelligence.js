import { api } from './api.js';

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const categorySelect = document.getElementById('category-select');
    const minProductsInput = document.getElementById('min-products');
    const applyFiltersBtn = document.getElementById('apply-filters');
    
    // Initialize the dashboard
    initializeDashboard();
    
    // Event listeners
    applyFiltersBtn.addEventListener('click', updateDashboard);
    
    async function initializeDashboard() {
        // Populate category dropdown
        populateCategories();
        
        // Load initial data
        updateDashboard();
    }
    
    function populateCategories() {
        // Use the predefined categories
        const categories = ['laptops', 'smartphones', 'wireless earbuds', 'smartwatches'];
        
        // Clear existing options
        categorySelect.innerHTML = '';
        
        // Populate the dropdown
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category.charAt(0).toUpperCase() + category.slice(1); // Capitalize first letter
            categorySelect.appendChild(option);
        });
        
        // Set laptops as default
        categorySelect.value = 'laptops';
    }
    
    async function updateDashboard() {
        const category = categorySelect.value;
        const minProducts = parseInt(minProductsInput.value) || 5;
        
        await Promise.all([
            updateBrandMetricsChart(category, minProducts),
            updateMarketShareChart(category)
        ]);
    }
    
    async function updateBrandMetricsChart(category, minProducts) {
        try {
            // Fetch data from API
            const params = { min_products: minProducts };
            if (category) params.category = category;
            
            // Using the avg-metrics endpoint as specified in your API docs
            const data = await api.get('/api/brand/avg-metrics', params);
            
            // Sort data by product count (descending)
            data.sort((a, b) => b.product_count - a.product_count);
            
            // Limit to top 10 brands for better visualization
            const topBrands = data.slice(0, 10);
            
            // Clear previous chart
            d3.select('#brand-metrics-chart').selectAll('*').remove();
            
            // Set up dimensions
            const margin = { top: 50, right: 30, bottom: 100, left: 60 };
            const width = 800 - margin.left - margin.right;
            const height = 500 - margin.top - margin.bottom;
            
            // Create SVG
            const svg = d3.select('#brand-metrics-chart')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', `translate(${margin.left},${margin.top})`);
            
            // Add title
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', -20)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .text(`Brand Performance Metrics - ${category.charAt(0).toUpperCase() + category.slice(1)}`);
            
            // Set up scales
            const x = d3.scaleBand()
                .domain(topBrands.map(d => d.brand))
                .range([0, width])
                .padding(0.3);
            
            const y = d3.scaleLinear()
                .domain([0, Math.max(
                    d3.max(topBrands, d => d.avg_price),
                    d3.max(topBrands, d => d.avg_rating * 20), // Scale ratings to be visible
                    d3.max(topBrands, d => d.avg_discount)
                ) * 1.1]) // Add 10% padding at the top
                .range([height, 0]);
            
            // Add X axis
            svg.append('g')
                .attr('transform', `translate(0,${height})`)
                .call(d3.axisBottom(x))
                .selectAll('text')
                .attr('transform', 'rotate(-45)')
                .style('text-anchor', 'end')
                .attr('dx', '-.8em')
                .attr('dy', '.15em');
            
            // Add Y axis
            svg.append('g')
                .call(d3.axisLeft(y));
            
            // Add Y axis label
            svg.append('text')
                .attr('transform', 'rotate(-90)')
                .attr('y', -40)
                .attr('x', -height / 2)
                .attr('text-anchor', 'middle')
                .text('Value');
            
            // Define colors
            const colors = {
                price: '#3498db',
                rating: '#f1c40f',
                discount: '#2ecc71'
            };
            
            // Add bars for average price
            svg.selectAll('.price-bar')
                .data(topBrands)
                .enter()
                .append('rect')
                .attr('class', 'price-bar')
                .attr('x', d => x(d.brand))
                .attr('y', d => y(d.avg_price))
                .attr('width', x.bandwidth() / 3)
                .attr('height', d => height - y(d.avg_price))
                .attr('fill', colors.price);
            
            // Add bars for average rating (scaled)
            svg.selectAll('.rating-bar')
                .data(topBrands)
                .enter()
                .append('rect')
                .attr('class', 'rating-bar')
                .attr('x', d => x(d.brand) + x.bandwidth() / 3)
                .attr('y', d => y(d.avg_rating * 20))
                .attr('width', x.bandwidth() / 3)
                .attr('height', d => height - y(d.avg_rating * 20))
                .attr('fill', colors.rating);
            
            // Add bars for average discount
            svg.selectAll('.discount-bar')
                .data(topBrands)
                .enter()
                .append('rect')
                .attr('class', 'discount-bar')
                .attr('x', d => x(d.brand) + 2 * x.bandwidth() / 3)
                .attr('y', d => y(d.avg_discount))
                .attr('width', x.bandwidth() / 3)
                .attr('height', d => height - y(d.avg_discount))
                .attr('fill', colors.discount);
            
            // Add legend
            const legend = svg.append('g')
                .attr('transform', `translate(${width - 200}, -40)`);
            
            const legendItems = [
                { label: 'Avg Price ($)', color: colors.price },
                { label: 'Avg Rating (0-5 × 20)', color: colors.rating },
                { label: 'Avg Discount (%)', color: colors.discount }
            ];
            
            legendItems.forEach((item, i) => {
                const legendGroup = legend.append('g')
                    .attr('transform', `translate(0, ${i * 20})`);
                
                legendGroup.append('rect')
                    .attr('width', 15)
                    .attr('height', 15)
                    .attr('fill', item.color);
                
                legendGroup.append('text')
                    .attr('x', 20)
                    .attr('y', 12)
                    .text(item.label)
                    .style('font-size', '12px');
            });
            
            // Add tooltips
            const tooltip = d3.select('body').append('div')
                .attr('class', 'tooltip')
                .style('opacity', 0)
                .style('position', 'absolute')
                .style('background-color', 'white')
                .style('border', '1px solid #ddd')
                .style('padding', '10px')
                .style('border-radius', '3px')
                .style('pointer-events', 'none');
            
            // Add tooltip functionality to bars
            svg.selectAll('rect')
                .on('mouseover', function(event, d) {
                    const barClass = this.getAttribute('class');
                    let value, label;
                    
                    if (barClass === 'price-bar') {
                        value = d.avg_price.toFixed(2);
                        label = 'Avg Price: $';
                    } else if (barClass === 'rating-bar') {
                        value = (d.avg_rating).toFixed(2);
                        label = 'Avg Rating: ';
                    } else {
                        value = d.avg_discount.toFixed(2);
                        label = 'Avg Discount: ';
                    }
                    
                    tooltip.transition()
                        .duration(200)
                        .style('opacity', .9);
                    
                    tooltip.html(`<strong>${d.brand}</strong><br>${label}${value}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                })
                .on('mouseout', function() {
                    tooltip.transition()
                        .duration(500)
                        .style('opacity', 0);
                });
            
        } catch (error) {
            console.error('Error updating brand metrics chart:', error);
            displayChartError('brand-metrics-chart', 'Failed to load brand metrics data');
        }
    }
    
    async function updateMarketShareChart(category) {
        try {
            // Fetch data from API
            const params = { category: category };
            
            // Using the market-share endpoint as specified in your API docs
            const data = await api.get('/api/brand/market-share', params);
            
            // Sort data by percentage (descending)
            data.sort((a, b) => b.percentage - a.percentage);
            
            // Group smaller brands into "Others" category for better visualization
            let topBrands = data.slice(0, 8); // Take top 8 brands
            
            if (data.length > 8) {
                const otherBrands = data.slice(8);
                const otherPercentage = otherBrands.reduce((sum, brand) => sum + brand.percentage, 0);
                const otherCount = otherBrands.reduce((sum, brand) => sum + brand.count, 0);
                
                topBrands.push({
                    brand: 'Others',
                    count: otherCount,
                    percentage: otherPercentage
                });
            }
            
            // Clear previous chart
            d3.select('#market-share-chart').selectAll('*').remove();
            
            // Set up dimensions
            const width = 500;
            const height = 500;
            const margin = 40;
            const radius = Math.min(width, height) / 2 - margin;
            
            // Create SVG
            const svg = d3.select('#market-share-chart')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${width / 2},${height / 2})`);
            
            // Add title
            svg.append('text')
                .attr('x', 0)
                .attr('y', -height / 2 + 20)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .text(`Market Share by Brand - ${category.charAt(0).toUpperCase() + category.slice(1)}`);
            
            // Define color scale
            const color = d3.scaleOrdinal()
                .domain(topBrands.map(d => d.brand))
                .range(d3.schemeCategory10);
            
            // Compute position of each group on the pie
            const pie = d3.pie()
                .value(d => d.percentage)
                .sort(null); // Do not sort, preserve the order
            
            const data_ready = pie(topBrands);
            
            // Build the pie chart
            const arcGenerator = d3.arc()
                .innerRadius(radius * 0.4) // For donut chart
                .outerRadius(radius);
            
            // Add arcs
            svg.selectAll('slices')
                .data(data_ready)
                .enter()
                .append('path')
                .attr('d', arcGenerator)
                .attr('fill', d => color(d.data.brand))
                .attr('stroke', 'white')
                .style('stroke-width', '2px')
                .style('opacity', 0.7);
            
            // Add labels
            const labelArc = d3.arc()
                .innerRadius(radius * 0.7)
                .outerRadius(radius * 0.7);
            
            svg.selectAll('labels')
                .data(data_ready)
                .enter()
                .append('text')
                .text(d => {
                    // Only show label if percentage is large enough
                    return d.data.percentage > 5 ? d.data.brand : '';
                })
                .attr('transform', d => `translate(${labelArc.centroid(d)})`)
                .style('text-anchor', 'middle')
                .style('font-size', '12px');
            
            // Add legend
            const legend = svg.append('g')
                .attr('transform', `translate(${radius + 20}, ${-radius})`);
            
            topBrands.forEach((brand, i) => {
                const legendRow = legend.append('g')
                    .attr('transform', `translate(0, ${i * 20})`);
                
                legendRow.append('rect')
                    .attr('width', 15)
                    .attr('height', 15)
                    .attr('fill', color(brand.brand));
                
                legendRow.append('text')
                    .attr('x', 20)
                    .attr('y', 12)
                    .text(`${brand.brand} (${brand.percentage.toFixed(1)}%)`)
                    .style('font-size', '12px');
            });
            
            // Add tooltips
            const tooltip = d3.select('body').append('div')
                .attr('class', 'tooltip')
                .style('opacity', 0)
                .style('position', 'absolute')
                .style('background-color', 'white')
                .style('border', '1px solid #ddd')
                .style('padding', '10px')
                .style('border-radius', '3px')
                .style('pointer-events', 'none');
            
            // Add tooltip functionality to pie slices
            svg.selectAll('path')
                .on('mouseover', function(event, d) {
                    tooltip.transition()
                        .duration(200)
                        .style('opacity', .9);
                    
                    tooltip.html(`<strong>${d.data.brand}</strong><br>
                                 Market Share: ${d.data.percentage.toFixed(2)}%<br>
                                 Products: ${d.data.count}`)
                                 .style('left', (event.pageX + 10) + 'px')
                                 .style('top', (event.pageY - 28) + 'px');
                         })
                         .on('mouseout', function() {
                             tooltip.transition()
                                 .duration(500)
                                 .style('opacity', 0);
                         });
                     
                 } catch (error) {
                     console.error('Error updating market share chart:', error);
                     displayChartError('market-share-chart', 'Failed to load market share data');
                 }
             }
             
             function displayChartError(chartId, message) {
                 // Clear the SVG
                 d3.select(`#${chartId}`).selectAll('*').remove();
                 
                 // Get SVG dimensions
                 const svg = d3.select(`#${chartId}`);
                 const width = +svg.attr('width') || 600;
                 const height = +svg.attr('height') || 400;
                 
                 // Add error message
                 svg.append('text')
                     .attr('x', width / 2)
                     .attr('y', height / 2)
                     .attr('text-anchor', 'middle')
                     .style('fill', '#d9534f')
                     .style('font-size', '16px')
                     .text(message);
             }
             
         });
         
