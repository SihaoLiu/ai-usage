//! Shared color palette for the monitor UI.

use ratatui::style::Color;

use crate::model_id::Vendor;

pub(super) const DIM: Color = Color::Indexed(245);
pub(super) const ACCENT: Color = Color::Indexed(51);
pub(super) const COL_PCT: Color = Color::Indexed(36);
pub(super) const ROW_PCT: Color = Color::Indexed(179);
pub(super) const SCALE_K: Color = Color::Indexed(108);
pub(super) const SCALE_M: Color = Color::Indexed(81);
pub(super) const SCALE_B: Color = Color::Indexed(214);
pub(super) const SCALE_T: Color = Color::Indexed(177);
pub(super) const TABLE_HEADER_BG: Color = Color::Indexed(236);
pub(super) const GROUP_BG: Color = Color::Indexed(235);
pub(super) const ZEBRA_BG: Color = Color::Indexed(233);
pub(super) const SUBTOTAL_BG: Color = Color::Indexed(234);
pub(super) const TOTAL_BG: Color = Color::Indexed(237);

pub(super) fn vendor_color(vendor: Vendor) -> Color {
    if vendor == Vendor::Unknown {
        return Color::Indexed(245);
    }
    const COLORS: [u8; 12] = [208, 255, 39, 49, 135, 44, 213, 75, 209, 220, 141, 214];
    Color::Indexed(COLORS[vendor.sort_rank() as usize % COLORS.len()])
}
