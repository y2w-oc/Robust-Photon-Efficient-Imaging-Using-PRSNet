function ConvertRGBDParsave(albedo_out, dist_out, intensity_out, dist_out_hr, albedo, dist, intensity, dist_hr)
    save(albedo_out, 'albedo');
    save(dist_out, 'dist');
    save(intensity_out, 'intensity');
    save(dist_out_hr, 'dist_hr');    

end

