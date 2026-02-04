import streamlit as st
import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations
import folium
from folium import plugins
from streamlit_folium import st_folium

class DroneBaseOptimizer:
    """
    P-median optimization for drone base station placement.
    """
    
    def __init__(self, blood_banks, health_facilities, operational_radius=80):
        self.blood_banks = np.array(blood_banks)
        self.health_facilities = np.array(health_facilities)
        self.operational_radius = operational_radius
        
        # Calculate distance matrix (in km)
        self.distance_matrix = self._calculate_distances()
        
        # Calculate coverage matrix (binary: 1 if within range, 0 otherwise)
        self.coverage_matrix = (self.distance_matrix <= operational_radius).astype(int)
    
    def _calculate_distances(self):
        """Calculate haversine distances between all blood banks and health facilities."""
        return cdist(self.blood_banks, self.health_facilities, 
                    metric=lambda u, v: self._haversine(u, v))
    
    def _haversine(self, coord1, coord2):
        """Calculate haversine distance between two coordinates in kilometers."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        r = 6371  # Earth's radius in kilometers
        return r * c
    
    def update_radius(self, new_radius):
        """Update operational radius and recalculate coverage matrix."""
        self.operational_radius = new_radius
        self.coverage_matrix = (self.distance_matrix <= new_radius).astype(int)
    
    def optimize_p_median(self, p, method='greedy'):
        """Find optimal p base stations to maximize coverage."""
        if method == 'greedy':
            return self._greedy_optimization(p)
        else:
            return self._exact_optimization(p)
    
    def _greedy_optimization(self, p):
        """Greedy algorithm: select base that covers most uncovered facilities."""
        selected = []
        covered = set()
        
        for _ in range(min(p, len(self.blood_banks))):
            best_base = None
            best_new_coverage = 0
            
            for i in range(len(self.blood_banks)):
                if i in selected:
                    continue
                
                # Find facilities covered by this base
                newly_covered = set(np.where(self.coverage_matrix[i] == 1)[0]) - covered
                
                if len(newly_covered) > best_new_coverage:
                    best_new_coverage = len(newly_covered)
                    best_base = i
            
            if best_base is not None:
                selected.append(best_base)
                covered.update(np.where(self.coverage_matrix[best_base] == 1)[0])
        
        return self._format_results(selected, covered)
    
    def _exact_optimization(self, p):
        """Exact optimization: try all combinations."""
        best_coverage = 0
        best_combination = None
        
        for combo in combinations(range(len(self.blood_banks)), p):
            covered = set()
            for base_idx in combo:
                covered.update(np.where(self.coverage_matrix[base_idx] == 1)[0])
            
            if len(covered) > best_coverage:
                best_coverage = len(covered)
                best_combination = combo
        
        covered = set()
        for base_idx in best_combination:
            covered.update(np.where(self.coverage_matrix[base_idx] == 1)[0])
        
        return self._format_results(list(best_combination), covered)
    
    def _format_results(self, selected_bases, covered_facilities):
        """Format optimization results."""
        return {
            'selected_bases': selected_bases,
            'covered_facilities': list(covered_facilities),
            'coverage_count': len(covered_facilities),
            'coverage_rate': len(covered_facilities) / len(self.health_facilities)
        }
    
    def create_folium_map(self, p, method='greedy'):
        """Create interactive Folium map showing drone coverage."""
        result = self.optimize_p_median(p, method=method)
        selected_bases = result['selected_bases']
        covered_facilities = set(result['covered_facilities'])
        
        # Calculate map center
        all_coords = np.vstack([self.blood_banks, self.health_facilities])
        center_lat = all_coords[:, 0].mean()
        center_lon = all_coords[:, 1].mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Create feature groups
        selected_bases_group = folium.FeatureGroup(name='Selected Drone Bases')
        unselected_bases_group = folium.FeatureGroup(name='Unselected Blood Banks')
        covered_facilities_group = folium.FeatureGroup(name='Covered Facilities')
        uncovered_facilities_group = folium.FeatureGroup(name='Uncovered Facilities')
        coverage_circles_group = folium.FeatureGroup(name=f'Coverage Radius ({self.operational_radius} km)')
        
        # Add unselected blood banks
        for idx, (lat, lon) in enumerate(self.blood_banks):
            if idx not in selected_bases:
                folium.Marker(
                    location=[lat, lon],
                    popup=f'Blood Bank {idx}<br>Not Selected',
                    icon=folium.Icon(color='gray', icon='home', prefix='fa'),
                    tooltip=f'Blood Bank {idx}'
                ).add_to(unselected_bases_group)
        
        # Add selected drone bases and coverage circles
        for idx in selected_bases:
            lat, lon = self.blood_banks[idx]
            
            # Add marker
            folium.Marker(
                location=[lat, lon],
                popup=f'<b>Drone Base {idx}</b><br>Selected Base Station<br>Radius: {self.operational_radius} km',
                icon=folium.Icon(color='red', icon='plane', prefix='fa'),
                tooltip=f'Drone Base {idx}'
            ).add_to(selected_bases_group)
            
            # Add coverage circle
            folium.Circle(
                location=[lat, lon],
                radius=self.operational_radius * 1000,  # Convert km to meters
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.1,
                opacity=0.5,
                popup=f'Coverage Area: {self.operational_radius} km radius',
                tooltip=f'Base {idx} Coverage'
            ).add_to(coverage_circles_group)
        
        # Add covered facilities
        for idx in covered_facilities:
            lat, lon = self.health_facilities[idx]
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=f'<b>Health Facility {idx}</b><br>Status: Covered',
                color='green',
                fillColor='green',
                fillOpacity=0.7,
                tooltip=f'Facility {idx} (Covered)'
            ).add_to(covered_facilities_group)
        
        # Add uncovered facilities
        uncovered = set(range(len(self.health_facilities))) - covered_facilities
        for idx in uncovered:
            lat, lon = self.health_facilities[idx]
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=f'<b>Health Facility {idx}</b><br>Status: NOT Covered',
                color='orange',
                fillColor='orange',
                fillOpacity=0.7,
                tooltip=f'Facility {idx} (Not Covered)'
            ).add_to(uncovered_facilities_group)
        
        # Add feature groups to map
        coverage_circles_group.add_to(m)
        unselected_bases_group.add_to(m)
        selected_bases_group.add_to(m)
        covered_facilities_group.add_to(m)
        uncovered_facilities_group.add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add minimap
        plugins.MiniMap().add_to(m)
        
        return m, result


# Streamlit App
def main():
    st.set_page_config(
        page_title="Drone Coverage Optimizer",
        page_icon="🚁",
        layout="wide"
    )
    
    st.title("🚁 Drone Base Station Coverage Optimizer")
    st.markdown("Optimize drone base placement to maximize health facility coverage")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Example data (replace with your actual data)
    # You can also add file upload functionality here
    if 'blood_banks' not in st.session_state:
        st.session_state.blood_banks = [
            [9.082, 8.675],   # Abuja region
            [6.465, 3.406],   # Lagos region
            [11.996, 8.525],  # Kano region
            [5.317, 7.384],   # Port Harcourt region
            [7.491, 4.551],   # Ibadan region
            [10.314, 9.845],  # Jos region
        ]
        
        st.session_state.health_facilities = [
            [9.050, 8.650], [9.100, 8.700], [9.200, 8.600],
            [6.450, 3.400], [6.500, 3.450], [6.550, 3.350],
            [12.000, 8.500], [11.950, 8.550], [11.900, 8.600],
            [5.300, 7.400], [5.350, 7.350], [5.250, 7.450],
            [7.500, 4.550], [7.450, 4.600], [7.550, 4.500],
            [10.300, 9.850], [10.350, 9.800], [10.250, 9.900],
        ]
    
    # Number of base stations slider
    max_bases = len(st.session_state.blood_banks)
    num_bases = st.sidebar.slider(
        "Number of Drone Bases",
        min_value=1,
        max_value=max_bases,
        value=min(3, max_bases),
        help="Select how many drone base stations to activate"
    )
    
    # Operational radius slider
    operational_radius = st.sidebar.slider(
        "Drone Operational Radius (km)",
        min_value=20,
        max_value=150,
        value=80,
        step=5,
        help="Maximum distance a drone can travel from base station"
    )
    
    # Optimization method
    method = st.sidebar.selectbox(
        "Optimization Method",
        options=['greedy', 'exact'],
        index=0,
        help="Greedy is faster; Exact finds optimal solution but slower for many bases"
    )
    
    # Add data info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.metric("Blood Banks Available", len(st.session_state.blood_banks))
    st.sidebar.metric("Health Facilities", len(st.session_state.health_facilities))
    
    # Initialize optimizer
    optimizer = DroneBaseOptimizer(
        blood_banks=st.session_state.blood_banks,
        health_facilities=st.session_state.health_facilities,
        operational_radius=operational_radius
    )
    
    # Create map and get results
    folium_map, result = optimizer.create_folium_map(num_bases, method=method)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Bases",
            f"{num_bases}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Facilities Covered",
            f"{result['coverage_count']}",
            delta=f"{result['coverage_count']} / {len(st.session_state.health_facilities)}"
        )
    
    with col3:
        st.metric(
            "Coverage Rate",
            f"{result['coverage_rate']:.1%}",
            delta=None
        )
    
    with col4:
        uncovered = len(st.session_state.health_facilities) - result['coverage_count']
        st.metric(
            "Uncovered Facilities",
            f"{uncovered}",
            delta=f"-{uncovered}" if uncovered > 0 else "0",
            delta_color="inverse"
        )
    
    # Display map
    st.markdown("---")
    st.markdown("### 🗺️ Coverage Map")
    st_folium(folium_map, width=1400, height=600)
    
    # Display selected bases
    st.markdown("---")
    st.markdown("### 📍 Selected Drone Base Stations")
    
    cols = st.columns(min(3, len(result['selected_bases'])))
    for i, base_idx in enumerate(result['selected_bases']):
        with cols[i % 3]:
            lat, lon = st.session_state.blood_banks[base_idx]
            st.info(f"**Base {base_idx}**  \nLat: {lat:.4f}  \nLon: {lon:.4f}")
    
    # Optional: Show coverage details
    with st.expander("📋 View Detailed Coverage Information"):
        covered_count = result['coverage_count']
        total_count = len(st.session_state.health_facilities)
        
        st.write(f"**Covered Facilities:** {covered_count} out of {total_count}")
        st.write(f"**Coverage Rate:** {result['coverage_rate']:.2%}")
        st.write(f"**Selected Base Indices:** {result['selected_bases']}")
        st.write(f"**Covered Facility Indices:** {sorted(result['covered_facilities'])}")
        
        uncovered_indices = sorted(set(range(total_count)) - set(result['covered_facilities']))
        if uncovered_indices:
            st.write(f"**Uncovered Facility Indices:** {uncovered_indices}")
        else:
            st.success("🎉 All facilities are covered!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with Streamlit | Optimization: P-Median Algorithm | Distance: Haversine Formula
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
