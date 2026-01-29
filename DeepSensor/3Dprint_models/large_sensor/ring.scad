/*  OpenSCAD input script
 *  Created using Claude AI with lot of instructions and fixes.
 *  
 *  TODO: smooth the springs both vertival and horizontal.
 * 
 */


// SCAD parameter "face number" to approximate ciylinders
$fn = 30;

// Parameters
height = 100;
outer_diameter = 116;
inner_diameter = 20;
wall_thickness = 5;
num_ribs = 3;
cut_diameter = outer_diameter - 30;
rib_height = 30;

// Spring parameters
spring_thickness = 2;
spring_height = 40;
spring_width = 30;
spring_extension = 4;

// Derived parameters
outer_radius = outer_diameter / 2;
inner_radius = inner_diameter / 2;
middle_radius = (outer_radius + inner_radius) / 2;
cut_radius = cut_diameter / 2;

// Calculated spring position
function chord_height(r, w) = r - sqrt(r*r - (w/2)*(w/2));
spring_chord_height = chord_height(outer_radius, spring_width);
spring_offset = outer_radius - spring_thickness/2 - spring_chord_height;

// Outer cylinder shell
module outer_shell() {
    difference() {
        cylinder(h = height, r = outer_radius, center = true);
        cylinder(h = height + 1, r = outer_radius - wall_thickness, center = true);
    }
}

// Inner cylinder shell
module inner_shell() {
    difference() {
        cylinder(h = height, r = inner_radius + wall_thickness, center = true);
        cylinder(h = height + 1, r = inner_radius, center = true);
    }
}

// Single rib
module rib() {
    cube([outer_radius - inner_radius, wall_thickness, height], center = true);
}

// Cut cylinder
module cut_cylinder() {
    cylinder(h = height - 2*rib_height, r = cut_radius, center = true);
}

// Single spring - box approximation
module spring_approx() {
    cube([spring_thickness, spring_width, spring_height], center = true);
}

// Simple planar spring (vertically bent)
module spring() {
    points = [
        [-spring_thickness/2, -spring_height/2],
        [spring_thickness/2, -spring_height/2],
        [spring_thickness/2 + spring_extension, 0],  // Bend point
        [spring_thickness/2, spring_height/2],
        [-spring_thickness/2, spring_height/2],
        [-spring_thickness/2 + spring_extension, 0]  // Bend point on the other side
    ];
    rotate([90, 0, 0])
    linear_extrude(height = spring_width, center = true) {
        polygon(points);
    }
}

// Main assembly
module main_assembly() {
    difference() {
        union() {
            outer_shell();
            inner_shell();
            for (i = [0:num_ribs-1]) {
                angle = i * 360 / num_ribs;
                rotate([0, 0, angle])
                translate([middle_radius, 0, 0])
                rib();
            }
        }
        cut_cylinder();
    }
    
    // Add springs
    for (i = [0:num_ribs-1]) {
        angle = i * 360 / num_ribs + 180 / num_ribs;  // Offset by half the angle between ribs
        rotate([0, 0, angle])
        translate([spring_offset, 0, height/2 + spring_height/2])
        spring();
    }
}

// Render the assembly
main_assembly();
