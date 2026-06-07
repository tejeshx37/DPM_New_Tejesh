use cgal::{curve::Curve, BoundaryId, PolygonSet, PolygonWithHoles};
use egui::{Context, Vec2b};
use egui_plot::{Line, Plot, PlotUi};
use std::hash::Hash;

pub fn plot(id_source: impl Hash) -> Plot {
    Plot::new(id_source)
        .data_aspect(1.0)
        .show_axes(Vec2b::FALSE)
}

pub fn plot_without_clutter(id_source: impl Hash) -> Plot {
    plot(id_source).show_grid(false).show_x(false).show_y(false)
}

pub fn default_transform(_: BoundaryId, ctx: &Context, line: Line) -> Line {
    line.color(super::on_primary_color(ctx))
}

pub fn plot_polygon_set<T>(ui: &mut PlotUi, polygon_set: &PolygonSet, transform: T)
where
    T: Fn(BoundaryId, &Context, Line) -> Line + Copy,
{
    polygon_set
        .polygon_with_holes()
        .iter()
        .flat_map(PolygonWithHoles::boundaries_iter)
        .map(|(id, curve)| {
            let line = match curve {
                Curve::Line(line) => Line::new(vec![
                    (line.end_points().start()).into(),
                    (line.end_points().end()).into(),
                ]),
                Curve::Ellipse(arc) => Line::new(arc.polyline().to_vec()),
            };
            (id, line)
        })
        .for_each(|(id, line)| ui.line(transform(id, ui.ctx(), line)))
}
