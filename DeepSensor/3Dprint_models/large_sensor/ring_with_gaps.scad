// OpenSCAD input script
// Created using Claude AI with lot of instructions and fixes.

// SCAD parameter "face number" to approximate ciylinders
$fn = 120;

// Parameters
main_height = 100;
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

// strip holes parameters
strip_width = 2;
strip_height = main_height - 2 * rib_height;


strip_angles_ = [0, 180, 300, 270, 350];
angle_zero = 20;
strip_angles = concat(
    [for (a = strip_angles_) angle_zero - a],
    [for (a = strip_angles_) angle_zero + a]);

// Derived parameters
outer_radius = outer_diameter / 2;
inner_radius = inner_diameter / 2;
middle_radius = (outer_radius + inner_radius) / 2;
cut_radius = cut_diameter / 2;

// Calculated spring position
function chord_height(r, w) = r - sqrt(r*r - (w/2)*(w/2));
spring_chord_height = chord_height(outer_radius, spring_width);
spring_offset = outer_radius - spring_thickness/2 - spring_chord_height;


module cylinder_shell(height, radius, thickness) {
    difference() {
        cylinder(h = height, r = radius, center = true);
        cylinder(h = height + 1, r = radius - thickness, center = true);
    }
}

// Outer cylinder shell
module outer_shell() {
    difference() {
        cylinder_shell(main_height, outer_radius, wall_thickness);
        for(angle = strip_angles) {
            rot_angle = (180 / num_ribs + angle) % 360;
            rotate([0, 0, rot_angle])
            translate([outer_radius, 0, 0])
            cube([3*wall_thickness, strip_width, strip_height], center = true);
        }    
    }
}

// Inner cylinder shell
module inner_shell() {
    cylinder_shell(main_height, inner_radius + wall_thickness, wall_thickness);
}

// Single rib
module rib() {
    cube([outer_radius - inner_radius, wall_thickness, main_height], center = true);
}

// Cut cylinder
module cut_cylinder() {
    cylinder(h = main_height - 2*rib_height, r = cut_radius, center = true);
}

// Single spring - box approximation
module spring_approx() {
    cube([spring_thickness, spring_width, spring_height], center = true);
}

module spring_vertical_cut() {
    L = spring_height;
    D = spring_extension;
    radius = (L*L + 4 * D*D) / (8 * D); 
    angle = 2 * acos( (radius - D) / radius );
    
    translate([-(radius - D), 0])
    intersection() {
        difference() {
            circle(radius);
            circle(radius - spring_thickness);
        }
        translate([radius/2, 0])
        square([radius, spring_height], center=true);
    }
}

// Simple planar spring (vertically bent)
module spring() {
    spring_angle = spring_width / outer_radius / 2 / PI * 180;
    rotate_extrude(angle=spring_angle, $fn=100) {
        translate([outer_radius, 0])
        //rotate([90, 0, 0])
        spring_vertical_cut();
    }
}

// Simple planar spring (vertically bent)
module spring_v() {
    points = [
        [-spring_thickness/2, -spring_height/2],
        [spring_thickness/2, -spring_height/2],
        [spring_thickness/2 + spring_extension, 0],  // Bend point
        [spring_thickness/2, spring_height/2],
        [-spring_thickness/2, spring_height/2],
        [-spring_thickness/2 + spring_extension, 0]  // Bend point on the other side
    ];
    
    spring_angle = spring_width / outer_radius / 2 / PI * 180;
    rotate_extrude(angle=spring_angle) {
        translate([outer_radius - spring_thickness/2, 0])
        //rotate([90, 0, 0])
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
        translate([0, 0, main_height/2 + spring_height/2])
        spring();
    }
}

// Render the assembly
main_assembly();
