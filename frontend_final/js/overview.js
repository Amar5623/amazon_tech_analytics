import { getPriceVsRating, getTopBrands, getDiscountVsRating, getReviewDistribution } from './api.js';

function renderScatterPlot(svgId, data, xKey, yKey, xLabel, yLabel) {
  const svg = d3.select(`#${svgId}`);
  const width = +svg.attr("width");
  const height = +svg.attr("height");
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  svg.selectAll("*").remove();

  const x = d3.scaleLinear()
    .domain([0, d3.max(data, d => d[xKey])])
    .range([0, innerWidth]);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d[yKey])])
    .range([innerHeight, 0]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("circle")
    .data(data)
    .enter()
    .append("circle")
    .attr("cx", d => x(d[xKey]))
    .attr("cy", d => y(d[yKey]))
    .attr("r", 4)
    .attr("fill", "steelblue");

  g.append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(x));

  g.append("g").call(d3.axisLeft(y));

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 5)
    .attr("text-anchor", "middle")
    .text(xLabel);

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 15)
    .attr("x", -height / 2)
    .attr("text-anchor", "middle")
    .text(yLabel);
}

function renderBarChart(svgId, data, labelKey, valueKey) {
  const svg = d3.select(`#${svgId}`);
  const width = +svg.attr("width");
  const height = +svg.attr("height");
  const margin = { top: 20, right: 20, bottom: 100, left: 40 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  svg.selectAll("*").remove();

  const x = d3.scaleBand()
    .domain(data.map(d => d[labelKey]))
    .range([0, innerWidth])
    .padding(0.2);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d[valueKey])])
    .range([innerHeight, 0]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("rect")
    .data(data)
    .enter()
    .append("rect")
    .attr("x", d => x(d[labelKey]))
    .attr("y", d => y(d[valueKey]))
    .attr("width", x.bandwidth())
    .attr("height", d => innerHeight - y(d[valueKey]))
    .attr("fill", "orange");

  g.append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(x))
    .selectAll("text")
    .attr("transform", "rotate(-45)")
    .style("text-anchor", "end");

  g.append("g").call(d3.axisLeft(y));
}

function renderHistogram(svgId, data) {
    const svg = d3.select(`#${svgId}`);
    const width = +svg.attr("width");
    const height = +svg.attr("height");
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
  
    svg.selectAll("*").remove();
  
    const x = d3.scaleLinear()
      .domain(d3.extent(data))
      .nice()
      .range([0, innerWidth]);
  
    const histogram = d3.histogram()
      .domain(x.domain())
      .thresholds(x.ticks(20));
  
    const bins = histogram(data);
  
    const y = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length)])
      .range([innerHeight, 0]);
  
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  
    g.selectAll("rect")
      .data(bins)
      .enter().append("rect")
      .attr("x", d => x(d.x0))
      .attr("y", d => y(d.length))
      .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 1))
      .attr("height", d => innerHeight - y(d.length))
      .attr("fill", "teal");
  
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x));
  
    g.append("g").call(d3.axisLeft(y));
  
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height - 5)
      .attr("text-anchor", "middle")
      .text("Review Count");
  
    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 15)
      .attr("x", -height / 2)
      .attr("text-anchor", "middle")
      .text("Frequency");
}

// Initialize charts
document.addEventListener('DOMContentLoaded', async function() {
  try {
    // Try to fetch real data
    const priceRatingData = await getPriceVsRating();
    const topBrandsData = await getTopBrands();
    const discountRatingData = await getDiscountVsRating();
    const reviewDistributionData = await getReviewDistribution();
    
    renderScatterPlot("price-rating", priceRatingData, "price", "rating", "Price", "Rating");
    renderBarChart("top-brands", topBrandsData, "brand", "count");
    renderScatterPlot("discount-rating", discountRatingData, "discount", "rating", "Discount (%)", "Rating");
    renderHistogram("review-distribution", reviewDistributionData);
  } catch (error) {
    console.error("Error fetching data from API, using mock data instead:", error);
    }
});
