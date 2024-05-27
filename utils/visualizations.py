import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import plotly.offline as offline
import utils.utils as utils
import torch
from PIL import Image
import os


def plot_cuts_iso(decode_points_func,
                  box_size=(2.5, 2.5, 2.5), max_n_eval_pts=1e6,
                  resolution=256, thres=0.0, imgs_per_cut=1, save_path=None) -> go.Figure:
    """ plot levelset at a certain cross section, assume inputs are centered
    Args:
        decode_points_func: A function to extract the SDF/occupancy logits of (N, 3) points
        box_size (List[float]): bounding box dimension
        max_n_eval_pts (int): max number of points to evaluate in one inference
        resolution (int): cross section resolution xy
        thres (float): levelset value
        imgs_per_cut (int): number of images for each cut (plotted in rows)
    Returns:
        a numpy array for the image
    """
    xmax, ymax, zmax = [b / 2 for b in box_size]
    xx, yy = np.meshgrid(np.linspace(-xmax, xmax, resolution),
                         np.linspace(-ymax, ymax, resolution))
    xx = xx.ravel()
    yy = yy.ravel()

    fig = make_subplots(rows=imgs_per_cut, cols=3,
                        subplot_titles=('xz', 'xy', 'yz'),
                        shared_xaxes='all', shared_yaxes='all',
                        vertical_spacing=0.01, horizontal_spacing=0.01,
                        )

    def _plot_cut(fig, idx, pos, decode_points_func, xmax, ymax, resolution):
        """ plot one cross section pos (3, N) """
        # evaluate points in serial
        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        feat = torch.zeros((field_input.shape[0], 1)).cuda()
        feat[:, 0] = 1
        values = decode_points_func(field_input).flatten()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            # np.save('xy.npy', values)
        values = values.reshape(resolution, resolution)
        contour_dict = dict(autocontour=False,
                            colorscale='RdBu',
                            reversescale=True,
                            contours=dict(
                                start=-0.1 + thres,
                                end=0.1 + thres,
                                size=0.01,
                                showlabels=True,  # show labels on contours
                                # labelfont=dict(  # label font properties
                                #     size=12,
                                #     color='white',
                                # )
                            ), )
        r_idx = idx // 3

        fig.add_trace(
            go.Contour(x=np.linspace(-xmax, xmax, resolution),
                       y=np.linspace(-ymax, ymax, resolution),
                       z=values,
                       **contour_dict
                       ),
            col=idx % 3 + 1, row=r_idx + 1  # 1-idx
        )

        fig.update_xaxes(
            range=[-xmax, xmax],  # sets the range of xaxis
            constrain="range",  # meanwhile compresses the xaxis by decreasing its "domain"
            col=idx % 3 + 1, row=r_idx + 1)
        fig.update_yaxes(
            range=[-ymax, ymax],
            col=idx % 3 + 1, row=r_idx + 1
        )

    steps = np.stack([np.linspace(-b / 2, b / 2, imgs_per_cut + 2)[1:-1] for b in box_size], axis=-1)
    for index in range(imgs_per_cut):
        position_cut = [np.vstack([xx, np.full(xx.shape[0], steps[index, 1]), yy]),
                        np.vstack([xx, yy, np.full(xx.shape[0], steps[index, 2])]),
                        np.vstack([np.full(xx.shape[0], steps[index, 0]), xx, yy]), ]
        _plot_cut(
            fig, index * 3, position_cut[0], decode_points_func, xmax, zmax, resolution)
        _plot_cut(
            fig, index * 3 + 1, position_cut[1], decode_points_func, xmax, ymax, resolution)
        _plot_cut(
            fig, index * 3 + 2, position_cut[2], decode_points_func, ymax, zmax, resolution)

    fig.update_layout(
        title='iso-surface',
        height=1200 * imgs_per_cut,
        width=1200 * 3,
        autosize=False,
        scene=dict(aspectratio=dict(x=1, y=1))
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


def plot_contours(x, y, z, colorscale='Geyser', fig=None):
    # plots a contour image given hight map matrix (2d array)
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], ),
                       yaxis=dict(side="left", range=[-1, 1], ),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='distance map', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Serif', size=24, color='red'))
                       )
    if fig is None:
        traces = []
    else:
        traces = list(fig.data)
    traces.append(go.Contour(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                             colorscale=colorscale,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 size=0.025,
                             ),
                             ))  # contour trace

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_scatter(x, y):
    layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                           yaxis=dict(range=[-1, 1], autorange=False),
                                                           aspectratio=dict(x=1, y=1)),
                       title=dict(text='scatter plot'))
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=16
        )
    ), layout=layout)

    fig.show()


def plot_contour_and_scatter_sal_training(clean_points, decoder, latent=None,
                                          example_idx=0, n_gt=None, n_pred=None,
                                          show_ax=True, show_bar=True, title_text='', colorscale='RdBu',
                                          res=256):
    # plot contour and scatter plot given input points, SAL decoder and the point's latent representation

    clean_points = clean_points[example_idx].detach().cpu().numpy()

    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()
    x, y, grid_points = utils.get_2d_grid_uniform(resolution=res, range=1.2, device=latent.device)
    if latent is not None:
        grid_points = torch.cat([latent[example_idx].expand(grid_points.shape[0], -1), grid_points], dim=1)
    z = decoder(grid_points).detach().cpu().numpy()

    traces = []
    # plot implicit function contour
    traces.append(go.Contour(x=x, y=y, z=z.reshape(x.shape[0], x.shape[0]),
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 size=0.025,
                             ), showscale=show_bar
                             ))  # contour trace
    # plot clean points and noisy points scatters
    traces.append(go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                             mode='markers', marker=dict(size=16, color=(0, 0, 0))))  # clean points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1],
                             line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    img = utils.plotly_fig2array(fig1)
    return img


def plot_contour_props(x_grid, y_grid, z_grid, clean_points,
                       z_diff,
                       example_idx=0, n_gt=None, n_pred=None,
                       show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                       nonmnfld_points=None, marker_size=10,
                       plot_second_derivs=False):
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []
    print(z_grid.min(), z_grid.max())

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 size=0.025,
                             ), showscale=show_bar
                             ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        num = 200
        clean_points_scatter = go.Scatter(x=clean_points[:num, 0], y=clean_points[:num, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter)  # clean points scatter

    if nonmnfld_points is not None:
        num = 200
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:num, 0], y=nonmnfld_points[:num, 1],
                                             mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter

    # plot normal vectors:
    if n_gt is not None:
        num = 20
        f = ff.create_quiver(clean_points[:num, 0], clean_points[:num, 1], n_gt[:num, 0], n_gt[:num, 1],
                             line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        num = 20
        f = ff.create_quiver(clean_points[:num, 0], clean_points[:num, 1], n_pred[:num, 0], n_pred[:num, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image")
    return dist_img


def plot_contour_div_props(x_grid, y_grid, z_grid, clean_points,
                           z_diff, eikonal_term, grid_div, grid_curl, hessian_det, eigs,
                           example_idx=0, n_gt=None, n_pred=None,
                           show_ax=False, show_bar=True, title_text='', colorscale='GnBu',
                           nonmnfld_points=None, marker_size=10,
                           plot_second_derivs=False, plot_hessian=True):
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
        if clean_points.shape[0] > 200:
            idx = np.random.choice(clean_points.shape[0], 200, replace=False)
            clean_points = clean_points[idx]
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []
    # plot clean points and noisy points scatters
    if clean_points is not None:
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                          mode='markers',
                                          # marker=dict(size=marker_size, color='rgba(0, 0, 0, 0.3)'))
                                          marker = dict(size=marker_size, color='rgba(255, 255, 255, 0.8)'))
        traces.append(clean_points_scatter)  # clean points scatter

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 # start=-.05,
                                 # end=.05,
                                 # size=0.005,
                                 start=-1,
                                 end=1,
                                 size=0.025,
                                 # showlabels=True,
                             ), showscale=show_bar,
                             # reversescale=True
                             ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(0, 0, 0)'], [1, 'rgb(0, 0, 0)']]))  # black bold zero line

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                             mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1],
                             line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image")
    # plot eikonal
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=eikonal_term,
                             colorscale=colorscale,
                             # reversescale=True,
                             # autocontour=True,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 size=0.025,
                             ), showscale=show_bar
                             ))  # contour trace
    fig3 = go.Figure(data=traces, layout=layout)
    offline.plot(fig3, auto_open=False)
    eikonal_img = utils.plotly_fig2array(fig3)
    print("Finished computing eikonal image")
    # plot z difference image
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_diff,
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-0.5,
                                 end=0.5,
                                 size=0.025,
                             ), showscale=show_bar,
                             reversescale=True
                             ))  # contour trace
    fig5 = go.Figure(data=traces, layout=layout)
    offline.plot(fig5, auto_open=False)
    z_diff_img = utils.plotly_fig2array(fig5)
    print("Finished computing gt distance image")
    if plot_second_derivs:
        # plot divergence
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_div, -10, 10),
                                 colorscale=colorscale,
                                 # autocontour=True,
                                 contours=dict(
                                     start=-10,
                                     end=10,
                                     size=1,
                                 ),
                                 showscale=show_bar
                                 ))  # contour trace
        fig4 = go.Figure(data=traces, layout=layout)
        offline.plot(fig4, auto_open=False)
        div_img = utils.plotly_fig2array(fig4)
        print("Finished divergence image")
        # plot curl
        # if clean_points is not None:
        #     traces = [clean_points_scatter]
        # traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_curl, -1e-4, 1e-4),
        #                            colorscale=colorscale,
        #                            # autocontour=True,
        #                            contours=dict(
        #                                start=-1e-4,
        #                                end=1e-4,
        #                                size=1e-5,
        #                            ),
        #                             showscale=show_bar
        #                            ))  # contour trace

        # fig2 = go.Figure(data=traces, layout=layout)
        # offline.plot(fig2, auto_open=False)
        # curl_img = utils.plotly_fig2array(fig2)
        # print("Finished computing curl image")
        curl_img = np.zeros_like(dist_img)
    else:
        # div_img, z_diff_img, curl_img = np.zeros_like(dist_img), np.zeros_like(dist_img), np.zeros_like(dist_img)
        div_img, curl_img = np.zeros_like(div_img), np.zeros_like(dist_img)
    if plot_hessian:
        # plot hessian
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Heatmap(x=x_grid, y=y_grid, z=hessian_det,
                                 colorscale=colorscale,
                                 showscale=show_bar,
                                 reversescale=True,
                                 ))
        # traces.append(go.Contour(x=x_grid, y=y_grid, z=hessian_det,
        #                          colorscale=colorscale,
        #                          contours=dict(
        #                              start=-1,
        #                              end=1,
        #                              size=0.025,
        #                          ), showscale=show_bar,
        #                          reversescale=True
        #                          ))
        fig5 = go.Figure(data=traces, layout=layout)
        offline.plot(fig5, auto_open=False)
        # hessian_det_img = utils.plotly_fig2array(fig5)
        print("Finished hessian_det image")
    else:
        hessian_det_img = None

    if eigs is not None:
        traces = list()
        if clean_points is not None:
            traces = [clean_points_scatter]
        eigs_max = eigs.max(axis=-1)
        eigs_min = eigs.min(axis=-1)
        # plot eigenvalues
        traces.append(go.Heatmap(x=x_grid, y=y_grid, z=eigs_max,
                                 colorscale=colorscale,
                                 showscale=show_bar,
                                 reversescale=True,
                                 text=eigs_max,
                                 ))
        fig6 = go.Figure(data=traces, layout=layout)
        offline.plot(fig6, auto_open=False)
        # eig_max_img = plotly_fig2array(fig4)

        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Heatmap(x=x_grid, y=y_grid, z=eigs_min,
                                 colorscale=colorscale,
                                 showscale=show_bar,
                                 reversescale=True,
                                 text=eigs_min,
                                 ))
        fig7 = go.Figure(data=traces, layout=layout)
        offline.plot(fig7, auto_open=False)
        fig7 = go.Figure(data=traces, layout=layout)
        offline.plot(fig7, auto_open=False)
    else:
        fig6 = None
        fig7 = None

    return dist_img, curl_img, eikonal_img, div_img, z_diff_img, fig5, fig6, fig7


def plot_init_contour_div_props(x_grid, y_grid, z_grid, clean_points,
                                z_diff, eikonal_term, grid_div, grid_curl,
                                example_idx=0, n_gt=None, n_pred=None,
                                show_ax=False, show_bar=True, title_text='', colorscale='RdBu',
                                nonmnfld_points=None, marker_size=10,
                                plot_second_derivs=False):
    import time
    t0 = time.time()
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []
    # plot clean points and noisy points scatters
    if clean_points is not None:
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter)  # clean points scatter

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             colorbar=dict(tickfont=dict(size=25)),
                             # contours_coloring='heatmap' # For smoothing
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 #    size=0.05, # Siren
                                 #    size=0.1, # geometric sine
                                 size=0.1,  # mfgi
                             ), showscale=show_bar
                             ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                             mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1],
                             line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image, {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    # plot eikonal
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=eikonal_term,
                             colorbar=dict(tickfont=dict(size=25)),
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 size=0.1,
                             ), showscale=show_bar
                             ))  # contour trace
    fig3 = go.Figure(data=traces, layout=layout)
    offline.plot(fig3, auto_open=False)
    eikonal_img = utils.plotly_fig2array(fig3)
    print("Finished computing eikonal image, {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    # plot z difference image
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_diff,
                             colorbar=dict(tickfont=dict(size=25)),
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-0.5,
                                 end=0.5,
                                 size=0.05,
                             ), showscale=show_bar
                             ))  # contour trace
    fig5 = go.Figure(data=traces, layout=layout)
    offline.plot(fig5, auto_open=False)
    z_diff_img = utils.plotly_fig2array(fig5)
    print("Finished computing gt distance image, {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    if plot_second_derivs:
        # plot divergence
        traces = []
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_div, -50, 50),
                                 colorbar=dict(tickfont=dict(size=25)),
                                 colorscale=colorscale,
                                 # autocontour=True,
                                 contours=dict(
                                     start=-50,
                                     end=50,
                                     size=1,
                                 ),
                                 contours_coloring='heatmap',
                                 #    showscale=show_bar
                                 ))  # contour trace
        fig4 = go.Figure(data=traces, layout=layout)
        offline.plot(fig4, auto_open=False)
        div_img = utils.plotly_fig2array(fig4)
        print("Finished divergence image, {:.2f}s".format(time.time() - t0))
        t0 = time.time()
        # # plot curl
        # traces = []
        # if clean_points is not None:
        #     traces = [clean_points_scatter]
        # traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_curl, -1e-4, 1e-4),
        #                         colorbar=dict(tickfont=dict(size=25)),
        #                            colorscale=colorscale,
        #                            # autocontour=True,
        #                            contours=dict(
        #                                start=-1e-4,
        #                                end=1e-4,
        #                                size=1e-5,
        #                            ),
        #                             showscale=show_bar
        #                            ))  # contour trace

        # fig2 = go.Figure(data=traces, layout=layout)
        # offline.plot(fig2, auto_open=False)
        # curl_img = utils.plotly_fig2array(fig2)
        # print("Finished computing curl image, {:.2f}s".format(time.time()-t0))
        # t0 = time.time()
        curl_img = None
    else:
        div_img, z_diff_img, curl_img = np.zeros_like(dist_img), np.zeros_like(dist_img), np.zeros_like(dist_img)
    return dist_img, curl_img, eikonal_img, div_img, z_diff_img


def plot_gt_contour_div_props(x_grid, y_grid, z_grid, clean_points,
                              z_diff, eikonal_term, grid_div, grid_curl,
                              example_idx=0, n_gt=None, n_pred=None,
                              show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                              nonmnfld_points=None, marker_size=10,
                              plot_second_derivs=False):
    import time
    t0 = time.time()
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-1,
                                 end=1,
                                 size=0.025,
                             ), showscale=show_bar
                             ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter)  # clean points scatter

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                             mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1],
                             line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image, {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    # plot eikonal
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=eikonal_term,
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=0,
                                 end=2,
                                 size=0.025,
                             ), showscale=show_bar
                             ))  # contour trace
    fig3 = go.Figure(data=traces, layout=layout)
    offline.plot(fig3, auto_open=False)
    eikonal_img = utils.plotly_fig2array(fig3)
    print("Finished computing eikonal image, {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    # plot z difference image
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_diff,
                             colorscale=colorscale,
                             # autocontour=True,
                             contours=dict(
                                 start=-0.5,
                                 end=0.5,
                                 size=0.025,
                             ), showscale=show_bar
                             ))  # contour trace
    fig5 = go.Figure(data=traces, layout=layout)
    offline.plot(fig5, auto_open=False)
    z_diff_img = utils.plotly_fig2array(fig5)
    print("Finished computing gt distance image, {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    if plot_second_derivs:
        # plot divergence
        traces = []
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_div, -50, 50),
                                 colorscale=colorscale,
                                 # autocontour=True,
                                 contours=dict(
                                     start=-50,
                                     end=50,
                                     size=1,
                                 ),
                                 contours_coloring='heatmap',
                                 #    showscale=show_bar
                                 ))  # contour trace
        fig4 = go.Figure(data=traces, layout=layout)
        offline.plot(fig4, auto_open=False)
        div_img = utils.plotly_fig2array(fig4)
        print("Finished divergence image, {:.2f}s".format(time.time() - t0))
        t0 = time.time()
        # plot curl
        traces = []
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_curl, -1e-4, 1e-4),
                                 colorscale=colorscale,
                                 # autocontour=True,
                                 contours=dict(
                                     start=-1e-4,
                                     end=1e-4,
                                     size=1e-5,
                                 ),
                                 showscale=show_bar
                                 ))  # contour trace

        fig2 = go.Figure(data=traces, layout=layout)
        offline.plot(fig2, auto_open=False)
        curl_img = utils.plotly_fig2array(fig2)
        print("Finished computing curl image, {:.2f}s".format(time.time() - t0))
        t0 = time.time()
        # curl_img = None
    else:
        div_img, z_diff_img, curl_img = np.zeros_like(dist_img), np.zeros_like(dist_img), np.zeros_like(dist_img)
    return dist_img, curl_img, eikonal_img, div_img, z_diff_img


def plot_paper_teaser_images(x_grid, y_grid, z_grid, clean_points, grid_normals, colorscale='Geyser', show_ax=False,
                             output_path='/mnt/3.5TB_WD/PycharmProjects/DiBS/sanitychecks/vis/'):
    os.makedirs(output_path, exist_ok=True)
    layout1 = go.Layout(width=1200, height=1200,
                        xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                        yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                        scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                   yaxis=dict(range=[-1, 1], autorange=False),
                                   aspectratio=dict(x=1, y=1)),
                        showlegend=False,
                        title=dict(text='', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                   font=dict(family='Sherif', size=24, color='red'))
                        )
    layout2 = go.Layout(width=1200, height=1200, plot_bgcolor='rgb(255, 255, 255)',
                        xaxis=dict(side="bottom", range=[-1, 1], showgrid=True, zeroline=show_ax, visible=True,
                                   gridcolor='LightGrey'),
                        yaxis=dict(side="left", range=[-1, 1], showgrid=True, zeroline=show_ax, visible=True,
                                   gridcolor='LightGrey'),
                        scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                   yaxis=dict(range=[-1, 1], autorange=False),
                                   aspectratio=dict(x=1, y=1)),
                        showlegend=False,
                        title=dict(text='', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                   font=dict(family='Sherif', size=24, color='red'))
                        )
    traces2 = []
    traces = []
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                             contours=dict(start=-3, end=3, size=0.1), showscale=False))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=5, dash="dash"),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace (zls)

    # plot clean points and noisy points scatters
    scatter_points = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                mode='markers', marker=dict(size=40, color='rgb(50, 50, 50)'))
    traces.append(scatter_points)
    traces2.append(scatter_points)
    traces.append(
        ff.create_quiver(x_grid, y_grid, grid_normals[:, 0], grid_normals[:, 1], line=dict(color='rgb(0, 0, 255)'),
                         scale=.04).data[0])

    fig1 = go.Figure(data=traces, layout=layout1)
    fig2 = go.Figure(data=traces2, layout=layout2)
    # fig2.show()
    offline.plot(fig1, auto_open=False)
    offline.plot(fig2, auto_open=False)
    img1 = utils.plotly_fig2array(fig1)
    img2 = utils.plotly_fig2array(fig2)
    im = Image.fromarray(img1)
    im.save(os.path.join(output_path, "teaser1.png"))
    im = Image.fromarray(img2)
    im.save(os.path.join(output_path, "teaser2.png"))


def plot_shape_data(x_grid, y_grid, z_grid, clean_points, n_gt=None, show_ax=True, show_bar=True,
                    title_text='', colorscale='Geyser', nonmnfld_points=None, divergence=None, grid_normals=None):
    # plot contour and scatter plot given input points

    traces = []
    # plot implicit function contour
    if divergence is None:
        traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                                 contours=dict(start=-1, end=1, size=0.025), showscale=show_bar))  # contour trace
    else:
        traces.append(go.Contour(x=x_grid, y=y_grid, z=divergence, colorscale=colorscale,
                                 contours=dict(start=-3, end=3, size=0.25), showscale=show_bar))  # contour trace

    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace

    # plot clean points and noisy points scatters
    clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                      mode='markers', marker=dict(size=16, color='rgb(0, 0, 0)'))
    traces.append(clean_points_scatter)  # clean points scatter

    if nonmnfld_points is not None:
        # plot clean points and noisy points scatters
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                             mode='markers', marker=dict(size=16, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # clean points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1],
                             line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])

    if grid_normals is not None:
        f = ff.create_quiver(x_grid, y_grid, grid_normals[:, 0], grid_normals[:, 1], line=dict(color='rgb(0, 0, 255)'),
                             scale=.01)
        traces.append(f.data[0])

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_sdf_indicator(points, x_grid, y_grid, sdf, title_text='', show_ax=False, output_path='./'):
    points = np.concatenate([points, points[0, :][None, :]], axis=0)
    indicator = -np.ones_like(sdf)
    indicator[sdf > 0] = 1

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=True,
                                  mirror=True, showline=True, linewidth=2, linecolor='black', showticklabels=False),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=True,
                                  mirror=True, showline=True, linewidth=2, linecolor='black', showticklabels=False),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red')),
                       paper_bgcolor='rgb(255,255,255)',
                       plot_bgcolor='rgb(255, 255, 255)'
                       )

    # plot clean points and noisy points scatters
    clean_points_scatter = go.Scatter(x=points[:, 0], y=points[:, 1],
                                      mode='markers', marker=dict(size=52, color='rgb(0, 0, 0)'))
    fig = go.Figure(data=clean_points_scatter, layout=layout)
    fig.write_image(os.path.join(output_path, "points.png"))
    fig.data = []
    line_plot = go.Scatter(x=points[:, 0], y=points[:, 1],
                           mode='markers+lines', marker=dict(size=52, color='rgb(0, 0, 0)'),
                           fill="toself", fillcolor='rgb(255,255,255)',
                           line=dict(width=6, color='rgb(0, 0, 0)'))

    fig.add_traces(line_plot)
    fig.write_image(os.path.join(output_path, "lines.png"))
    fig.data = []

    sdf_trace = go.Contour(x=x_grid, y=y_grid, z=sdf, colorscale='Geyser',
                           contours=dict(start=-1, end=1, size=0.025), showscale=False)
    sdf_zls_trace = go.Contour(x=x_grid, y=y_grid, z=sdf,
                               contours=dict(start=0, end=0, coloring='lines'),
                               line=dict(width=5),
                               showscale=False,
                               colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']])  # contour trace
    fig.add_traces([sdf_trace, sdf_zls_trace, clean_points_scatter])
    fig.write_image(os.path.join(output_path, "sdf.png"))
    fig.data = []

    indicator_trace = go.Contour(x=x_grid, y=y_grid, z=indicator, colorscale='Geyser',
                                 contours=dict(start=-1, end=1, size=0.025), showscale=False)
    indicator_zls_trace = go.Contour(x=x_grid, y=y_grid, z=indicator,
                                     contours=dict(start=0, end=0, coloring='lines'),
                                     line=dict(width=5),
                                     showscale=False,
                                     colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']])  # contour trace
    fig.add_traces([indicator_trace, indicator_zls_trace, clean_points_scatter])
    fig.write_image(os.path.join(output_path, "indicator.png"))

    # fig = go.Figure(data=traces, layout=layout)
    # fig.show()


def plot_distance_map(x_grid, y_grid, z_grid, title_text='', colorscale='Geyser', show_ax=True):
    # plot contour and scatter plot given input points

    traces = []
    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                             contours=dict(start=-1, end=1, size=0.025), showscale=True))
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0, end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_points(points, title_txt='circle', example_idx=0, show_ax=True):
    points = points[example_idx]

    traces = []
    traces.append(go.Scatter(x=points[:, 0], y=points[:, 1],
                             mode='markers', marker=dict(size=16, color='rgb(0, 0, 0)')))  # points scatter

    # plot jet
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='Shape: ' + title_txt, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_mesh(mesh_trace, mesh=None, output_ply_path=None, show_ax=False, title_txt='', show=True):
    # plot mesh and export ply file for mesh

    # export mesh
    if not (mesh is None or output_ply_path is None):
        mesh.export(output_ply_path, vertex_normal=True)

    # plot mesh
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                  yaxis=dict(range=[-2, 2], autorange=False),
                                  zaxis=dict(range=[-2, 2], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='Shape: ' + title_txt, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    fig = go.Figure(data=mesh_trace, layout=layout)
    if show:
        fig.show()


def plot_sdf_surface(x, y, z, show=True, show_ax=True, title_txt=''):
    # plot mesh
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                  yaxis=dict(range=[-2, 2], autorange=False),
                                  zaxis=dict(range=[-2, 2], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_txt, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    surf_trace = go.Surface(z=z, x=x, y=y)
    fig = go.Figure(data=surf_trace, layout=layout)
    if show:
        fig.show()

    return fig
