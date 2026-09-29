from svgutils.compose import SVG, Figure, Panel, Text

svg_scale = 1.334  # set this to 1.334 for Inkscape, 1.0 otherwise

# Panel letters in Helvetica Neue, 12pt, Medium
kwargs_text = {"size": "10pt", "font": "Arial", "weight": "800"}
kwargs_text_normal = {"size": "8pt", "font": "Arial"}

f = Figure(
    "598",
    "128",
    Panel(
        SVG("./panels/panela.svg").scale(svg_scale).move(-5, -13),
        Text("a", 0, 7.5, **kwargs_text),
    ).move(0, 0),
    Panel(
        SVG("./panels/panelb.svg").scale(svg_scale).move(0, -2),
        # Text("b", 135, 2.0, **kwargs_text),
    ).move(175, 0),
    Panel(
        SVG("./panels/panelc.svg").scale(svg_scale).move(-5, -5),
        Text("b", 0, 0, **kwargs_text),
    ).move(323, 10),
    # Panel(
    #     SVG("./panels/paneld.svg").scale(svg_scale).move(430, -3),
    #     Text("c", 405, 2.0, **kwargs_text),
    # ).move(85, 8),
)

f.save("final.svg")
