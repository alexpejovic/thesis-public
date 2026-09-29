from svgutils.compose import SVG, Figure, Panel

svg_scale = 1.334

f = Figure(
    "205",
    "176",
    Panel(
        SVG("./panels/panela.svg").scale(svg_scale).move(0, 0),
    ).move(-9, -9),
)

f.save("final.svg")
