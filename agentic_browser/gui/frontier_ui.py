"""
Frontier UI - Mission Control Interface for Agentic Browser.

A high-density "Glass Cockpit" interface with:
- Neural Stream (left): Scrolling agent thoughts/actions
- Viewport (center): Browser screenshots, terminal, graph
- State Panel (right): Real-time JSON state visualization
"""

import sys
import json
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QSplitter, QFrame,
    QTabWidget, QScrollArea, QTreeWidget, QTreeWidgetItem,
    QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QStatusBar, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, QSize, QRectF, QPointF
from PySide6.QtGui import (
    QFont, QColor, QIcon, QPalette, QTextCursor, QPen, QBrush,
    QPixmap, QPainter,
)


# --- CONFIGURATION & STYLING ---
THEME_BG_DARK = "#0f1115"       # Deepest background
THEME_BG_PANEL = "#161b22"      # Panel background
THEME_BORDER = "#30363d"        # Border color
THEME_ACCENT = "#58a6ff"        # Bright Blue (Actions)
THEME_SUCCESS = "#238636"       # Green (Success)
THEME_WARNING = "#d29922"       # Yellow/Orange (Thinking)
THEME_ERROR = "#f85149"         # Red (Errors)
THEME_TEXT_MAIN = "#c9d1d9"     # Main text
THEME_TEXT_DIM = "#8b949e"      # Dimmed text
THEME_PURPLE = "#d2a8ff"        # Router/Supervisor color
FONT_MONO = "JetBrains Mono"    # Preferred mono font
FONT_UI = "Inter"               # Preferred UI font

# Agent node colors for graph
NODE_COLORS = {
    "planner": "#f778ba",       # Pink
    "supervisor": "#d2a8ff",    # Purple
    "browser": "#58a6ff",       # Blue
    "research": "#7ee787",      # Green
    "os": "#ffa657",            # Orange
    "code": "#ff7b72",          # Red
    "data": "#79c0ff",          # Light blue
    "network": "#a5d6ff",       # Cyan
    "media": "#ffb4a2",         # Salmon
    "automation": "#ffd166",    # Yellow
    "retrospective": "#b392f0", # Light purple
}

STYLESHEET = f"""
QMainWindow {{ background-color: {THEME_BG_DARK}; }}
QWidget {{ color: {THEME_TEXT_MAIN}; font-family: '{FONT_UI}', sans-serif; }}
QFrame {{ border: none; }}

/* Splitters */
QSplitter::handle {{ background-color: {THEME_BORDER}; width: 1px; }}

/* Panels */
.PanelFrame {{ 
    background-color: {THEME_BG_PANEL}; 
    border: 1px solid {THEME_BORDER}; 
    border-radius: 6px; 
}}

/* Tabs */
QTabWidget::pane {{ border: 1px solid {THEME_BORDER}; background: {THEME_BG_PANEL}; }}
QTabBar::tab {{
    background: {THEME_BG_DARK};
    color: {THEME_TEXT_DIM};
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{ background: {THEME_BG_PANEL}; color: {THEME_ACCENT}; border-top: 2px solid {THEME_ACCENT}; }}

/* Inputs */
QLineEdit {{
    background-color: #0d1117;
    border: 1px solid {THEME_BORDER};
    border-radius: 4px;
    padding: 8px;
    color: {THEME_TEXT_MAIN};
    font-family: '{FONT_UI}';
}}
QLineEdit:focus {{ border: 1px solid {THEME_ACCENT}; }}

/* Buttons */
QPushButton.Primary {{
    background-color: {THEME_SUCCESS};
    color: white;
    border: 1px solid rgba(240, 246, 252, 0.1);
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}}
QPushButton.Primary:hover {{ background-color: #2ea043; }}

QPushButton.Danger {{
    background-color: {THEME_ERROR};
    color: white;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}}
QPushButton.Danger:hover {{ background-color: #da3633; }}

/* Tree Widget (State) */
QTreeWidget {{
    background-color: {THEME_BG_PANEL};
    border: none;
    font-family: '{FONT_MONO}', monospace;
    font-size: 11px;
}}
QHeaderView::section {{
    background-color: {THEME_BG_DARK};
    border: none;
    border-bottom: 1px solid {THEME_BORDER};
    padding: 4px;
}}

/* Scroll areas */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: {THEME_BG_PANEL};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {THEME_BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Text edits */
QTextEdit {{
    background-color: #0d1117;
    border: none;
    font-family: '{FONT_MONO}', monospace;
}}
"""


# --- UI COMPONENTS ---

class NeuralCard(QFrame):
    """A single block in the Neural Stream (Thought, Action, or System)."""
    
    def __init__(self, data: dict):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 8, 10, 8)
        self.layout.setSpacing(4)
        
        # Header Row
        header_layout = QHBoxLayout()
        
        # Type Indicator
        self.type_lbl = QLabel()
        self.type_lbl.setFont(QFont(FONT_MONO, 8, QFont.Weight.Bold))
        
        event_type = data.get("type", "system")
        agent = data.get("agent", "")
        
        if event_type == "thought":
            self.type_lbl.setText(f"💭 {agent.upper()} THOUGHT" if agent else "THOUGHT")
            self.type_lbl.setStyleSheet(f"color: {THEME_WARNING};")
            self.setStyleSheet(f"background-color: #1c1c1c; border-left: 3px solid {THEME_WARNING}; border-radius: 4px;")
        elif event_type == "action":
            self.type_lbl.setText(f"⚡ {agent.upper()} ACTION" if agent else "ACTION")
            self.type_lbl.setStyleSheet(f"color: {THEME_ACCENT};")
            self.setStyleSheet(f"background-color: #121d2b; border-left: 3px solid {THEME_ACCENT}; border-radius: 4px;")
        elif event_type == "router":
            self.type_lbl.setText("🔀 ROUTER")
            self.type_lbl.setStyleSheet(f"color: {THEME_PURPLE};")
            self.setStyleSheet(f"background-color: #161b22; border-left: 3px solid {THEME_PURPLE}; border-radius: 4px;")
        elif event_type == "success":
            self.type_lbl.setText("✅ COMPLETE")
            self.type_lbl.setStyleSheet(f"color: {THEME_SUCCESS};")
            self.setStyleSheet(f"background-color: #0f2416; border-left: 3px solid {THEME_SUCCESS}; border-radius: 4px;")
        elif event_type == "error":
            self.type_lbl.setText("❌ ERROR")
            self.type_lbl.setStyleSheet(f"color: {THEME_ERROR};")
            self.setStyleSheet(f"background-color: #2d1b1b; border-left: 3px solid {THEME_ERROR}; border-radius: 4px;")
        else:
            self.type_lbl.setText("ℹ️ SYSTEM")
            self.type_lbl.setStyleSheet(f"color: {THEME_TEXT_DIM};")
            self.setStyleSheet(f"background-color: {THEME_BG_PANEL}; border-left: 3px solid {THEME_BORDER}; border-radius: 4px;")

        # Timestamp
        time_lbl = QLabel(data.get("timestamp", ""))
        time_lbl.setStyleSheet(f"color: {THEME_TEXT_DIM}; font-size: 10px;")
        
        header_layout.addWidget(self.type_lbl)
        header_layout.addStretch()
        header_layout.addWidget(time_lbl)
        
        # Content
        content = data.get("content", "")
        content_lbl = QLabel(content)
        content_lbl.setWordWrap(True)
        content_lbl.setStyleSheet(f"color: {THEME_TEXT_MAIN}; font-size: 12px;")
        content_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        self.layout.addLayout(header_layout)
        self.layout.addWidget(content_lbl)


class NeuralStream(QWidget):
    """Left Pane: The scrolling history of agent thoughts/actions."""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.container_layout.setSpacing(8)
        self.container_layout.setContentsMargins(4, 4, 4, 4)
        
        self.scroll.setWidget(self.container)
        self.layout.addWidget(self.scroll)
        
    def add_event(self, data: dict):
        """Add a new event card to the stream."""
        card = NeuralCard(data)
        self.container_layout.addWidget(card)
        # Auto scroll to bottom
        QTimer.singleShot(10, self._scroll_to_bottom)
    
    def _scroll_to_bottom(self):
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        )
    
    def clear(self):
        """Clear all events from the stream."""
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class GraphVisualizerWidget(QGraphicsView):
    """Visual representation of the agent graph with real-time highlighting."""
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet(f"background: {THEME_BG_DARK}; border: none;")
        
        # Node positions (circular layout)
        self.nodes = {}
        self.edges = []
        self._current_node = None
        
        self._setup_graph()
    
    def _setup_graph(self):
        """Set up the initial graph visualization."""
        # Node positions in a semi-circular layout
        node_names = [
            "planner", "supervisor", "browser", "research", "os",
            "code", "data", "network", "media", "automation"
        ]
        
        center_x, center_y = 200, 150
        radius = 120
        
        import math
        for i, name in enumerate(node_names):
            angle = math.pi * (i / (len(node_names) - 1))  # Semi-circle
            x = center_x + radius * math.cos(angle)
            y = center_y - radius * math.sin(angle) + 50
            
            # Special position for planner and supervisor
            if name == "planner":
                x, y = center_x - 50, 30
            elif name == "supervisor":
                x, y = center_x + 50, 30
            
            self._add_node(name, x, y)
        
        # Add edges (supervisor -> all workers, planner -> supervisor)
        self._add_edge("planner", "supervisor")
        for name in node_names[2:]:  # Skip planner and supervisor
            self._add_edge("supervisor", name)
    
    def _add_node(self, name: str, x: float, y: float):
        """Add a node to the graph."""
        color = NODE_COLORS.get(name, THEME_ACCENT)
        
        # Circle
        ellipse = QGraphicsEllipseItem(x - 18, y - 18, 36, 36)
        ellipse.setPen(QPen(QColor(color), 2))
        ellipse.setBrush(QBrush(QColor(THEME_BG_DARK)))
        self.scene.addItem(ellipse)
        
        # Label
        label = QGraphicsTextItem(name[:3].upper())
        label.setDefaultTextColor(QColor(color))
        label.setFont(QFont(FONT_MONO, 7, QFont.Weight.Bold))
        label.setPos(x - 12, y - 8)
        self.scene.addItem(label)
        
        self.nodes[name] = {"ellipse": ellipse, "label": label, "x": x, "y": y, "color": color}
    
    def _add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            return
        
        from_data = self.nodes[from_node]
        to_data = self.nodes[to_node]
        
        line = QGraphicsLineItem(
            from_data["x"], from_data["y"],
            to_data["x"], to_data["y"]
        )
        line.setPen(QPen(QColor(THEME_BORDER), 1, Qt.PenStyle.DashLine))
        self.scene.addItem(line)
        self.edges.append(line)
    
    def highlight_node(self, node_name: str):
        """Highlight the currently active node."""
        # Reset previous node
        if self._current_node and self._current_node in self.nodes:
            prev = self.nodes[self._current_node]
            prev["ellipse"].setBrush(QBrush(QColor(THEME_BG_DARK)))
            prev["ellipse"].setPen(QPen(QColor(prev["color"]), 2))
        
        # Highlight new node
        if node_name in self.nodes:
            node = self.nodes[node_name]
            node["ellipse"].setBrush(QBrush(QColor(node["color"])))
            node["ellipse"].setPen(QPen(QColor("#ffffff"), 3))
            self._current_node = node_name


class StatePanel(QTreeWidget):
    """Right Pane: Real-time visualization of JSON state."""
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Key", "Value"])
        self.setColumnWidth(0, 140)
        self.setAlternatingRowColors(False)
        self.setRootIsDecorated(True)

    def update_state(self, state_dict: dict, parent=None):
        """Update the tree with new state data."""
        if parent is None:
            self.clear()
            parent = self.invisibleRootItem()
            
        for key, value in state_dict.items():
            # Skip internal keys
            if key.startswith("_"):
                continue
                
            item = QTreeWidgetItem(parent)
            item.setText(0, str(key))
            
            if isinstance(value, dict):
                item.setText(1, "{...}")
                self.update_state(value, item)
                item.setExpanded(True)
            elif isinstance(value, list):
                item.setText(1, f"[{len(value)} items]")
                # Add list items as children
                for i, v in enumerate(value[:10]):  # Limit to 10 items
                    child = QTreeWidgetItem(item)
                    child.setText(0, f"[{i}]")
                    child.setText(1, str(v)[:100])
            else:
                text = str(value)[:100]
                item.setText(1, text)
                item.setForeground(1, QColor(THEME_ACCENT))


class BrowserViewport(QLabel):
    """Center browser tab showing screenshots from Playwright."""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: black; color: {THEME_TEXT_DIM};")
        self.setText("Waiting for browser session...")
        self.setMinimumSize(400, 300)
    
    def update_screenshot(self, screenshot_bytes: bytes):
        """Update the viewport with a new screenshot.
        
        Optimized to prevent GUI blocking:
        - Uses FastTransformation instead of SmoothTransformation
        - Limits update frequency (throttled in agent_thread)
        - Wrapped in try/except to prevent crashes
        """
        try:
            pixmap = QPixmap()
            if not pixmap.loadFromData(screenshot_bytes):
                return  # Skip if load fails
            
            # Scale to fit while maintaining aspect ratio
            # Using FastTransformation (not Smooth) to reduce GUI blocking
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation  # Much faster than Smooth
            )
            self.setPixmap(scaled)
        except Exception as e:
            print(f"[GUI] Screenshot update error: {e}")
    
    def show_url(self, url: str):
        """Show URL when no screenshot is available."""
        self.setText(f"🌐 {url}\n\n(Screenshot capture in progress...)")


# --- MAIN WINDOW ---

class MissionControlWindow(QMainWindow):
    """Main Mission Control window with three-pane layout."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agentic AiDEN // Mission Control")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)
        
        # State
        self._agent_thread = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the main window UI."""
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # --- HEADER ---
        header = QHBoxLayout()
        
        title = QLabel("AGENTIC AiDEN")
        title.setFont(QFont(FONT_MONO, 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {THEME_TEXT_MAIN}; letter-spacing: 1px;")
        
        self.status_badge = QLabel(" ● READY ")
        self.status_badge.setStyleSheet(
            f"background: {THEME_SUCCESS}; color: white; border-radius: 4px; "
            f"font-size: 10px; font-weight: bold; padding: 2px 6px;"
        )
        
        header.addWidget(title)
        header.addWidget(self.status_badge)
        header.addStretch()
        
        # Cost Ticker
        self.cost_lbl = QLabel("COST: $0.0000 | TOKENS: 0")
        self.cost_lbl.setFont(QFont(FONT_MONO, 11))
        self.cost_lbl.setStyleSheet(f"color: {THEME_TEXT_DIM};")
        header.addWidget(self.cost_lbl)
        
        # Settings Button
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.setFont(QFont(FONT_UI, 10))
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: {THEME_BG_PANEL};
                border: 1px solid {THEME_BORDER};
                border-radius: 4px;
                padding: 4px 12px;
                color: {THEME_TEXT_MAIN};
            }}
            QPushButton:hover {{
                background: {THEME_BORDER};
            }}
        """)
        self.settings_btn.clicked.connect(self._on_settings)
        header.addWidget(self.settings_btn)
        
        # Costs Button
        self.costs_btn = QPushButton("💰 Costs")
        self.costs_btn.setFont(QFont(FONT_UI, 10))
        self.costs_btn.setStyleSheet(f"""
            QPushButton {{
                background: {THEME_BG_PANEL};
                border: 1px solid {THEME_BORDER};
                border-radius: 4px;
                padding: 4px 12px;
                color: {THEME_TEXT_MAIN};
            }}
            QPushButton:hover {{
                background: {THEME_BORDER};
            }}
        """)
        self.costs_btn.clicked.connect(self._on_costs)
        header.addWidget(self.costs_btn)
        
        main_layout.addLayout(header)
        
        # --- GOAL INPUT ---
        goal_layout = QHBoxLayout()
        
        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("Enter your goal (e.g., 'Research the best Python web frameworks')...")
        self.goal_input.setFont(QFont(FONT_UI, 12))
        
        self.run_btn = QPushButton("▶ RUN")
        self.run_btn.setProperty("class", "Primary")
        self.run_btn.setFont(QFont(FONT_UI, 11, QFont.Weight.Bold))
        self.run_btn.setMinimumWidth(100)
        
        self.stop_btn = QPushButton("⬛ STOP")
        self.stop_btn.setProperty("class", "Danger")
        self.stop_btn.setFont(QFont(FONT_UI, 11, QFont.Weight.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(100)
        
        goal_layout.addWidget(self.goal_input, 1)
        goal_layout.addWidget(self.run_btn)
        goal_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(goal_layout)
        main_layout.addWidget(self._create_separator())
        
        # --- SPLITTER LAYOUT ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        
        # 1. LEFT PANE: NEURAL STREAM
        left_pane = QFrame()
        left_pane.setProperty("class", "PanelFrame")
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        l_header = QLabel("  📡 NEURAL STREAM")
        l_header.setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        l_header.setStyleSheet(
            f"background: {THEME_BG_PANEL}; padding: 8px; "
            f"border-bottom: 1px solid {THEME_BORDER}; color: {THEME_TEXT_DIM};"
        )
        left_layout.addWidget(l_header)
        
        self.stream = NeuralStream()
        left_layout.addWidget(self.stream)
        
        # Steering Input
        steering_container = QWidget()
        steering_container.setStyleSheet(
            f"background: {THEME_BG_PANEL}; border-top: 1px solid {THEME_BORDER};"
        )
        sc_layout = QVBoxLayout(steering_container)
        sc_layout.setContentsMargins(8, 8, 8, 8)
        
        steering_lbl = QLabel("⚡ LIVE STEERING:")
        steering_lbl.setFont(QFont(FONT_MONO, 9))
        steering_lbl.setStyleSheet(f"color: {THEME_WARNING};")
        
        self.steering_input = QLineEdit()
        self.steering_input.setPlaceholderText("Inject context (e.g., 'Check Amazon too')...")
        self.steering_input.setEnabled(False)
        
        sc_layout.addWidget(steering_lbl)
        sc_layout.addWidget(self.steering_input)
        
        left_layout.addWidget(steering_container)
        
        # 2. CENTER PANE: VIEWPORT
        center_pane = QTabWidget()
        center_pane.setProperty("class", "PanelFrame")
        
        # Browser View
        self.browser_viewport = BrowserViewport()
        
        # Terminal View
        self.terminal_view = QTextEdit()
        self.terminal_view.setReadOnly(True)
        self.terminal_view.setStyleSheet(
            f"background: #0d1117; color: #7ee787; "
            f"font-family: '{FONT_MONO}'; font-size: 11px;"
        )
        self.terminal_view.setText("$ ready\n")
        
        # Graph Visualizer
        self.graph_view = GraphVisualizerWidget()
        
        center_pane.addTab(self.browser_viewport, "🌐 BROWSER")
        center_pane.addTab(self.terminal_view, "💻 TERMINAL")
        center_pane.addTab(self.graph_view, "🔀 GRAPH")
        
        # 3. RIGHT PANE: STATE & MEMORY
        right_pane = QFrame()
        right_pane.setProperty("class", "PanelFrame")
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        r_header = QLabel("  📊 AGENT STATE")
        r_header.setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        r_header.setStyleSheet(
            f"background: {THEME_BG_PANEL}; padding: 8px; "
            f"border-bottom: 1px solid {THEME_BORDER}; color: {THEME_TEXT_DIM};"
        )
        right_layout.addWidget(r_header)
        
        self.state_tree = StatePanel()
        right_layout.addWidget(self.state_tree)
        
        # Add to Splitter
        splitter.addWidget(left_pane)
        splitter.addWidget(center_pane)
        splitter.addWidget(right_pane)
        
        # Set Ratios (25% | 50% | 25%)
        splitter.setSizes([350, 700, 350])
        
        main_layout.addWidget(splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to run")
    
    def _create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {THEME_BORDER};")
        line.setFixedHeight(1)
        return line
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.goal_input.returnPressed.connect(self._on_run)
        self.steering_input.returnPressed.connect(self._send_steering)
    
    def _on_run(self):
        """Start agent execution."""
        goal = self.goal_input.text().strip()
        if not goal:
            QMessageBox.warning(self, "No Goal", "Please enter a goal first.")
            return
        
        # Import here to avoid circular imports
        from .agent_thread import AgentThread
        from ..config import AgentConfig
        from ..settings_store import SettingsStore
        
        # Get config from settings
        store = SettingsStore()
        settings = store.settings
        provider_config = settings.get_provider_config()
        
        config = AgentConfig(
            goal=goal,
            model=provider_config.effective_model,
            model_endpoint=provider_config.endpoint,
            api_key=provider_config.api_key,
            headless=settings.headless,
            vision_mode=settings.vision_mode,
            auto_approve=settings.auto_approve,
        )
        
        # Clear previous run
        self.stream.clear()
        self.state_tree.clear()
        self.terminal_view.clear()
        
        # Update UI
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.steering_input.setEnabled(True)
        self.goal_input.setEnabled(False)
        self.status_badge.setText(" ● RUNNING ")
        self.status_badge.setStyleSheet(
            f"background: {THEME_ACCENT}; color: white; border-radius: 4px; "
            f"font-size: 10px; font-weight: bold; padding: 2px 6px;"
        )
        
        # Create and start agent thread
        self._agent_thread = AgentThread(config, goal, settings.max_steps)
        self._agent_thread.signal_log.connect(self.stream.add_event)
        self._agent_thread.signal_state.connect(self._update_state)
        self._agent_thread.signal_usage.connect(self._update_usage)
        self._agent_thread.signal_node.connect(self.graph_view.highlight_node)
        self._agent_thread.signal_screenshot.connect(self.browser_viewport.update_screenshot)
        self._agent_thread.signal_complete.connect(self._on_complete)
        self._agent_thread.signal_error.connect(self._on_error)
        self._agent_thread.signal_terminal.connect(self._append_terminal)
        self._agent_thread.start()
        
        # Note: Screenshots are captured internally by agent thread
        # to avoid cross-thread Playwright/greenlet errors
        
        self.stream.add_event({
            "type": "system",
            "content": f"🎯 Goal: {goal}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    
    def _on_stop(self):
        """Stop agent execution."""
        if self._agent_thread:
            self._agent_thread.abort()
        
        self.stream.add_event({
            "type": "system",
            "content": "⏹️ Stopping agent...",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    
    def _send_steering(self):
        """Send steering input to agent."""
        text = self.steering_input.text().strip()
        if text and self._agent_thread:
            self._agent_thread.inject_steering(text)
            self.steering_input.clear()
    
    def _update_state(self, state: dict):
        """Update state panel with new state."""
        # Filter to interesting keys
        display_state = {
            k: v for k, v in state.items()
            if k in ("goal", "current_url", "extracted_data", "step_count", 
                    "visited_urls", "current_node", "error")
        }
        self.state_tree.update_state(display_state)
        
        # Update browser viewport URL if present
        if "current_url" in state:
            self.browser_viewport.show_url(state["current_url"])
    
    def _update_usage(self, usage: dict):
        """Update token/cost display."""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_cost = usage.get("total_cost", 0.0)
        
        self.cost_lbl.setText(
            f"COST: ${total_cost:.4f} | IN: {input_tokens:,} | OUT: {output_tokens:,}"
        )
    
    def _on_complete(self, success: bool, result: str):
        """Handle agent completion."""
        self._reset_ui()
        
        if success:
            self.status_badge.setText(" ● COMPLETE ")
            self.status_badge.setStyleSheet(
                f"background: {THEME_SUCCESS}; color: white; border-radius: 4px; "
                f"font-size: 10px; font-weight: bold; padding: 2px 6px;"
            )
            self.stream.add_event({
                "type": "success",
                "content": result[:500] if result else "Task completed successfully",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
        else:
            self.status_badge.setText(" ● STOPPED ")
            self.status_badge.setStyleSheet(
                f"background: {THEME_WARNING}; color: white; border-radius: 4px; "
                f"font-size: 10px; font-weight: bold; padding: 2px 6px;"
            )
    
    def _on_error(self, error: str):
        """Handle agent error."""
        self._reset_ui()
        
        self.status_badge.setText(" ● ERROR ")
        self.status_badge.setStyleSheet(
            f"background: {THEME_ERROR}; color: white; border-radius: 4px; "
            f"font-size: 10px; font-weight: bold; padding: 2px 6px;"
        )
        
        self.stream.add_event({
            "type": "error",
            "content": error,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    
    def _reset_ui(self):
        """Reset UI to ready state."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.steering_input.setEnabled(False)
        self.goal_input.setEnabled(True)
    
    def _on_settings(self):
        """Open settings dialog."""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.stream.add_event({
                "type": "system",
                "content": "⚙️ Settings updated",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
    
    def _on_costs(self):
        """Open costs dialog showing API usage breakdown."""
        from .cost_dialog import CostDialog
        # Get current usage from agent thread if available
        usage_data = {}
        if self._agent_thread:
            # Try to get accumulated usage
            usage_data = getattr(self._agent_thread, '_accumulated_usage', {})
        
        dialog = CostDialog(self, usage_data)
        dialog.exec()
    
    def _append_terminal(self, text: str):
        """Append text to the terminal view."""
        self.terminal_view.append(text.rstrip())
        # Auto-scroll to bottom
        scrollbar = self.terminal_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle window close - ensure proper cleanup."""
        # Force cleanup of agent thread and browser
        if self._agent_thread and self._agent_thread.isRunning():
            self._agent_thread.abort()
            
            # Wait a bit for graceful shutdown
            if not self._agent_thread.wait(1500):
                # Force terminate if still running
                print("[GUI] Force terminating agent thread...")
                self._agent_thread.terminate()
                self._agent_thread.wait(500)
        
        # Cleanup tool registry to close any open browsers
        try:
            from ..graph.tool_registry import ToolRegistry
            registry = ToolRegistry.get_instance()
            registry.clear()
        except Exception as e:
            print(f"[GUI] Cleanup error: {e}")
        
        event.accept()


# --- ENTRY POINT ---
def main():
    """Launch Mission Control UI."""
    app = QApplication(sys.argv)
    app.setFont(QFont(FONT_UI, 9))
    
    window = MissionControlWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
