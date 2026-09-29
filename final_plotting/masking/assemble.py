from svgutils.compose import SVG, Figure, Panel, Text

svg_scale = 1.334  # set this to 1.334 for Inkscape, 1.0 otherwise

# Panel letters in Helvetica Neue, 12pt, Medium
kwargs_text = {"size": "10pt", "font": "Arial", "weight": "800"}
kwargs_text_normal = {"size": "8pt", "font": "Arial"}

f = Figure(
    "580",
    "228",
    Panel(
        SVG("./panels/panela.svg").scale(svg_scale).move(0, 0),
        Text("a", 5, 2.0, **kwargs_text),
    ).move(-5, 40),
    Panel(
        SVG("./panels/panelb.svg").scale(svg_scale).move(0, 0),
        Text("b", 5, 2.0, **kwargs_text),
    ).move(220, 8),
    Panel(
        SVG("./panels/panelc.svg").scale(svg_scale).move(0, 0),
        Text("c", 5, 2.0, **kwargs_text),
    ).move(220, 125),
    Panel(
        SVG("./panels/paneld.svg").scale(svg_scale).move(0, 0),
        Text("d", 5, 2.0, **kwargs_text),
    ).move(465, 8),
    Panel(
        SVG("./panels/panele.svg").scale(svg_scale).move(0, 0),
        Text("e", 5, 2.0, **kwargs_text),
    ).move(462, 125),
)

f.save("final.svg")
