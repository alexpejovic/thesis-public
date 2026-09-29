from svgutils.compose import SVG, Figure, Panel, Text

svg_scale = 1.334  # set this to 1.334 for Inkscape, 1.0 otherwise

# Panel letters in Helvetica Neue, 12pt, Medium
kwargs_text = {"size": "10pt", "font": "Arial", "weight": "800"}
kwargs_text_normal = {"size": "8pt", "font": "Arial"}

f = Figure(
    "600",
    "211",
    Panel(
        SVG("./panels/panela.svg").scale(svg_scale).move(0, 0),
        Text("a", 0, 2.0, **kwargs_text),
    ).move(15, 8),
    Panel(
        SVG("./panels/panelb.svg").scale(svg_scale).move(205, 0),
        Text("b", 220, 2.0, **kwargs_text),
    ).move(85, 8),
)

f.save("final.svg")
