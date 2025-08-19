from pyvis.network import Network
import os

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph

    def _generate_legend_html(self, color_map):
        """Generates HTML and CSS for a legend based on the topic-color map."""
        legend_html = '<div class="legend">'
        legend_html += '<h4>Topic Legend</h4>'
        legend_html += '<ul>'
        for topic, color in color_map.items():
            legend_html += f'<li><span class="color-box" style="background-color:{color};"></span>{topic}</li>'
        legend_html += '</ul></div>'

        style_html = '''
        <style>
        .legend {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border: 1px solid #d3d3d3;
            border-radius: 8px;
            font-family: sans-serif;
            font-size: 14px;
            z-index: 1000;
            max-height: 200px;
            overflow-y: auto;
        }
        .legend h4 {
            margin-top: 0;
            margin-bottom: 10px;
        }
        .legend ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .legend li {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        .legend .color-box {
            width: 15px;
            height: 15px;
            display: inline-block;
            margin-right: 8px;
            border: 1px solid #000;
        }
        </style>
        '''
        return style_html + legend_html

    def visualize(self, output_path='data/output/graph.html'):
        net = Network(notebook=True, cdn_resources='in_line', height="750px", width="100%")

        # 1. Create consistent color mapping for topics
        topics = sorted(list(set(d.get('topic') for n, d in self.graph.nodes(data=True) if d.get('topic'))))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        topic_color_map = {topic: colors[i % len(colors)] for i, topic in enumerate(topics)}

        # 2. Pre-calculate triplet-to-topic mapping for coloring
        triplet_to_topic = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get('type') == 'contains_triplet':
                chunk_node = self.graph.nodes[u]
                triplet_node_id = v
                if chunk_node.get('type') == 'chunk':
                    triplet_to_topic[triplet_node_id] = chunk_node.get('topic')

        # 3. Add nodes with topic-based coloring and metadata
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'N/A')
            title_text = f"ID: {node_id}\nType: {node_type}\nContent: {node_data.get('content', 'N/A')}"
            node_color = '#D3D3D3'  # Default color for unclassified/other nodes
            node_label = str(node_id)

            topic = node_data.get('topic')
            if node_type == "chunk":
                if topic in topic_color_map:
                    node_color = topic_color_map[topic]
                title_text += f"\nFile: {node_data.get('filename', 'N/A')}\nTopic: {topic}"

            elif node_type == "triplet":
                topic = triplet_to_topic.get(node_id)
                if topic in topic_color_map:
                    node_color = topic_color_map[topic]
                node_label = node_data.get('content', '')
                title_text += f"\nTopic: {topic}"

            elif node_type == "topic":
                node_color = topic_color_map.get(node_data.get('content'), '#8A2BE2')
                node_label = node_data.get('label', '')
                title_text = f"Type: {node_type}\nTopic: {node_data.get('content', 'N/A')}"

            net.add_node(node_id, title=title_text, label=node_label, color=node_color)

        # 4. Add edges (no changes needed here)
        for u, v, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get('type', 'N/A')
            color = '#97C2FC'
            if edge_type == 'semantic_similarity':
                color = '#FF0000'
                title_text = f"Type: {edge_type}\nWeight: {edge_data.get('weight', 'N/A'):.2f}"
            elif edge_type == 'contains_triplet':
                color = '#00FF00'
                title_text = f"Type: {edge_type}"
            elif edge_type == 'has_topic':
                color = '#FFA500'
                title_text = f"Type: {edge_type}"
            else:
                title_text = f"Type: {edge_type}"
            net.add_edge(u, v, color=color, title=title_text)

        # 5. Generate graph and inject the legend
        net.show(output_path)
        with open(output_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            legend_html = self._generate_legend_html(topic_color_map)
            # Inject legend before the closing body tag
            content = content.replace('</body>', legend_html + '</body>')
            f.seek(0)
            f.write(content)
            f.truncate()

        print(f"Interactive graph visualization with legend saved to {output_path}")