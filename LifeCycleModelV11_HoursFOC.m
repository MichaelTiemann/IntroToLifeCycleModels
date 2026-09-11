function h_star = LifeCycleModelV11_HoursFOC(aprime, a, z, e, w, psi, eta, agej, Jr, kappa_j, r)

if agej >= Jr
    h_star = 0 .* a; % Could initialize via zeros
else
    x = (1 + r) .* a - aprime;
    w_eff = w .* kappa_j .* z .* e;
    discriminant = x.^2 + (4 .* (w_eff.^2)) ./ psi;
    h_star = (-x + sqrt(discriminant)) ./ (2 .* w_eff);
    h_star = min(1, max(0, h_star));
end


end
